from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import asyncio
import whisper
import torchaudio
import numpy as np
import re
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass

import tempfile
import os
import warnings
from typing import List
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
# import ffmpeg

# Config
CHUNK_DURATION = 120
IGNORE_WIN_WARNINGS = True
MAX_FILES_PER_REQUEST = 25  # Maximum number of files per request
CONCURRENT_TRANSCRIPTIONS = 2  # It takes about 6GB per instance on Windows, maybe better on Linux
HALLUCINATION_COMPRESSION = 4.1  # 2.1
NO_SPEECH = 0.7  # 0.3
FP16 = False  # True

# ----- ADD: batching config -----
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES, N_SAMPLES
FRAME_DURATION = 30          # батчируем по 30 секунд — нативный размер окна Whisper
BATCH_SIZE = 8               # подбирай по VRAM (6–8 обычно ок для turbo)
MAX_MEL_FRAMES = N_FRAMES       # эталонная длина мел-спектра (обычно 3000 на 30с)
N_SAMPLES_30S = N_SAMPLES       # эталонная длина аудио 30с (обычно 480_000)
from whisper import DecodingOptions
from whisper.decoding import DecodingTask
# --------------------------------

app = FastAPI()

# Suppress specific warnings if needed
if IGNORE_WIN_WARNINGS:
    warnings.filterwarnings(
        "ignore",
        message=".*Torch was not compiled with flash attention.*",
        category=UserWarning,
        module=".*whisper\.model"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Failed to launch Triton kernels.*",
        category=UserWarning,
        module=".*whisper\.timing"
    )

# Global variable to hold the model in worker processes
model = None

# Test Batching
# ----- ADD: helper для детекта языка один раз на канал -----
def _detect_language_code_for_channel(mel_tensor):
    """
    mel_tensor: torch.Tensor [80, T] на device модели
    Возвращает двухбуквенный код ('en', 'ru', ...) для DecodingOptions.
    """
    global model
    with torch.no_grad():
        probs = model.detect_language(mel_tensor.unsqueeze(0))[0]
    return max(probs, key=probs.get)
# ------------------------------------------------------------


# ----- ADD: батчевый декодер 30-сек фреймов -----
def _decode_mel_batch(mel_batch, language_code, suppress_tokens, fp16):
    global model
    opts = DecodingOptions(
        language=language_code,
        task="transcribe",
        fp16=fp16,
        suppress_tokens=suppress_tokens or None,
        without_timestamps=False,
        temperature=0.0,
    )
    task = DecodingTask(model, opts)
    with torch.inference_mode():
        results = task.run(mel_batch)
    return results
# ------------------------------------------------

def _approx_words(text: str, start: float, end: float):
    text = (text or "").strip()
    if not text or end <= start:
        return []
    tokens = [t for t in re.findall(r"\w+|[^\w\s]+", text) if t.strip()]
    n = len(tokens)
    if n == 0:
        return []
    dt = (end - start) / n
    t = start
    words = []
    for tok in tokens:
        words.append({"word": tok, "start": t, "end": t + dt})
        t += dt
    return words

OVERLAP_S = 5.0  # seconds of overlap between 30s windows

def _stitch_segments(prev_segs, cur_segs, overlap_start, overlap_end):
    """
    Merge two consecutive windows with [overlap_start, overlap_end] in absolute time.
    Heuristic: prefer segments/words from the later window inside the overlap,
    unless the earlier one has higher avg_logprob.
    """
    # split prev into keep/ovl
    keep_prev, ovl_prev = [], []
    for s in prev_segs:
        if s["end"] <= overlap_start: keep_prev.append(s)
        elif s["start"] < overlap_end: ovl_prev.append(s)
        else: keep_prev.append(s)  # should be rare; non-overlapped tail

    # compute avg logprob per side (if available)
    def avg_lp(segs):
        vals = [seg.get("avg_logprob") for seg in segs if seg.get("avg_logprob") is not None]
        return sum(vals)/len(vals) if vals else None

    lp_prev = avg_lp(ovl_prev)
    lp_cur  = avg_lp([s for s in cur_segs if s["start"] < overlap_end])

    prefer_prev = (lp_prev is not None and lp_cur is not None and lp_prev >= lp_cur)

    if prefer_prev:
        # drop overlapping part of current window
        trimmed_cur = [s for s in cur_segs if s["start"] >= overlap_end]
        return keep_prev + ovl_prev + trimmed_cur
    else:
        # drop overlapping tail of previous window
        keep_prev = [s for s in keep_prev if s["end"] <= overlap_start]
        trimmed_prev = [s for s in prev_segs if s["end"] <= overlap_start]
        return trimmed_prev + cur_segs

def _offset_segment_times(seg, offset):
    seg = dict(seg)
    seg["start"] += offset
    seg["end"]   += offset
    if "words" in seg and seg["words"]:
        for w in seg["words"]:
            w["start"] += offset
            w["end"]   += offset
    return seg

def transcribe_file_process_batched(file_path, file_name, language="auto"):
    """
    Quality-preserving chunker:
      - 30s windows with 5s overlap
      - condition_on_previous_text=True
      - word_timestamps=True
      - stitches overlaps
    """
    print(f"{datetime.now()} - [QUALITY] Starting transcription for {file_name} (lang={language})")
    global model
    try:
        if model is None:
            model = whisper.load_model("turbo", device="cuda")
            print(f"{datetime.now()} - [QUALITY] Model (re)loaded in PID {os.getpid()}")

        # Load & resample
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        num_channels = waveform.shape[0]
        all_transcriptions = {}

        def transcribe_channel(wave_1d, ch_name):
            window = int(30.0 * sample_rate)
            step   = int((30.0 - OVERLAP_S) * sample_rate)  # 25s step → 5s overlap
            nsamp  = wave_1d.shape[0]

            # window starts
            starts = list(range(0, max(1, nsamp - window + 1), step))
            if starts[-1] + window < nsamp:
                starts.append(nsamp - window)  # tail

            results_all = []
            prev_abs_segments = None
            for idx, start_sample in enumerate(starts):
                end_sample = min(start_sample + window, nsamp)
                chunk = wave_1d[start_sample:end_sample].numpy()
                start_time = start_sample / sample_rate

                # Decode with *real* word timestamps
                res = model.transcribe(
                    audio=chunk,
                    language=(None if language.lower() == "auto" else language),
                    task="transcribe",
                    temperature=(0.0, 0.1, 0.2),
                    no_speech_threshold=NO_SPEECH,
                    suppress_tokens=[
                        50365, 2933, 8893, 403, 1635, 10461, 40653,
                        413, 4775, 51, 284, 89, 453, 51864, 50366,
                        8567, 1435, 21403, 5627, 15363, 17781, 485,
                        51863
                    ],
                    condition_on_previous_text=True,    # carry context
                    word_timestamps=True,               # crucial
                    compression_ratio_hallucination_threshold=HALLUCINATION_COMPRESSION,
                    fp16=FP16,
                )

                # Collect absolute-time segments (with words)
                cur_abs_segments = []
                for seg in res.get("segments", []):
                    seg_abs = _offset_segment_times(seg, start_time)
                    cur_abs_segments.append(seg_abs)

                if prev_abs_segments is None:
                    prev_abs_segments = cur_abs_segments
                else:
                    # stitch with overlap window [start_time, start_time + OVERLAP_S]
                    stitched = _stitch_segments(
                        prev_abs_segments, cur_abs_segments,
                        overlap_start=start_time,
                        overlap_end=start_time + OVERLAP_S
                    )
                    prev_abs_segments = stitched

            return prev_abs_segments or []

        # Per channel
        for ch in range(num_channels):
            ch_name = f"Speaker {ch}"
            segments = transcribe_channel(waveform[ch].contiguous(), ch_name)

            # normalize schema: add id & fill words if Whisper didn’t provide
            out, next_id = [], 0
            for seg in segments:
                seg = dict(seg)
                seg["id"] = next_id; next_id += 1
                if "words" not in seg or not seg["words"]:
                    # fallback (rare with word_timestamps=True)
                    seg["words"] = _approx_words(seg.get("text",""), seg["start"], seg["end"])
                out.append(seg)

            all_transcriptions[ch_name] = out

        print(f"{datetime.now()} - [QUALITY] {file_name}: done")
        return file_name, all_transcriptions

    except Exception as e:
        err = f"{file_name}: QUALITY error: {e}"
        print(f"{datetime.now()} - {err}")
        return file_name, {"error": err}




def worker_init():
    """
    Function to initialize the model in each worker process.
    This function will be called when each process starts.
    """
    global model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    model = whisper.load_model("turbo", device="cuda")
    print(f"{datetime.now()} - [Worker {os.getpid()}] Model loaded.")

def transcribe_file_process(file_path, file_name, language="auto"):
    print(f"{datetime.now()} - Starting transcription for {file_name} with language parameter: {language}")
    global model
    try:
        if model is None:
            print(f"{datetime.now()} - Model not loaded in process {os.getpid()}; loading now.")
            model = whisper.load_model("turbo", device="cuda")
        
        # Convert "auto" to None for Whisper's language auto-detection.
        transcribe_language = None if language.lower() == "auto" else language

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        num_channels = waveform.shape[0]
        print(f"{datetime.now()} - {file_name}: Loaded audio with sample rate {sample_rate} Hz and {num_channels} channel(s).")

        # Resample to 16000 Hz if necessary
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
            print(f"{datetime.now()} - {file_name}: Resampled audio to {sample_rate} Hz.")

        chunk_duration = CHUNK_DURATION  # seconds
        chunk_samples = int(chunk_duration * sample_rate)
        all_transcriptions = {}

        if num_channels == 2:
            # Process each channel separately.
            for channel_idx in range(num_channels):
                channel_waveform = waveform[channel_idx].unsqueeze(0)
                channel_name = f"Speaker {channel_idx}"
                num_samples = channel_waveform.shape[1]
                num_chunks = (num_samples + chunk_samples - 1) // chunk_samples
                transcriptions = []
                print(f"{datetime.now()} - {file_name}: Processing {num_chunks} chunks for {channel_name}.")
                for i in range(num_chunks):
                    start_sample = i * chunk_samples
                    end_sample = min((i + 1) * chunk_samples, num_samples)
                    chunk_waveform = channel_waveform[:, start_sample:end_sample]
                    start_time = start_sample / sample_rate
                    chunk_numpy = chunk_waveform.squeeze().numpy()

                    print(f"{datetime.now()} - {file_name}: Transcribing chunk {i+1}/{num_chunks} for {channel_name} (start_time: {start_time:.2f}s).")
                    result = model.transcribe(
                        audio=chunk_numpy,
                        language=transcribe_language,
                        task="transcribe",
                        temperature=(0.0, 0.1),
                        no_speech_threshold=NO_SPEECH,
                        suppress_tokens=[
                            50365, 2933, 8893, 403, 1635, 10461, 40653,
                            413, 4775, 51, 284, 89, 453, 51864, 50366,
                            8567, 1435, 21403, 5627, 15363, 17781, 485,
                            51863
                        ],
                        condition_on_previous_text=False,
                        word_timestamps=True,
                        compression_ratio_hallucination_threshold=HALLUCINATION_COMPRESSION,
                        fp16=FP16,
                    )
                    print(f"{datetime.now()} - {file_name}: Chunk {i+1} produced {len(result['segments'])} segment(s).")
                    for segment in result["segments"]:
                        segment['start'] += start_time
                        segment['end'] += start_time
                        transcriptions.append(segment)

                all_transcriptions[channel_name] = transcriptions

            print(f"{datetime.now()} - {file_name}: Completed transcription for stereo file.")
            return file_name, all_transcriptions

        elif num_channels == 1:  # Mono file
            transcriptions = []
            num_samples = waveform.shape[1]
            num_chunks = (num_samples + chunk_samples - 1) // chunk_samples
            print(f"{datetime.now()} - {file_name}: Processing mono file in {num_chunks} chunk(s).")
            next_segment_id = 0  # Unique ID counter

            for i in range(num_chunks):
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, num_samples)
                chunk_waveform = waveform[:, start_sample:end_sample]
                start_time = start_sample / sample_rate
                chunk_numpy = chunk_waveform.squeeze().numpy()

                print(f"{datetime.now()} - {file_name}: Transcribing chunk {i+1}/{num_chunks} (start_time: {start_time:.2f}s).")
                result = model.transcribe(
                    audio=chunk_numpy,
                    language=transcribe_language,
                    task="transcribe",
                    temperature=(0.0, 0.1, 0.15, 0.2),
                    no_speech_threshold=NO_SPEECH,
                    suppress_tokens=[
                        50365, 2933, 8893, 403, 1635, 10461, 40653,
                        413, 4775, 51, 284, 89, 453, 51864, 50366,
                        8567, 1435, 21403, 5627, 15363, 17781, 485,
                        51863
                    ],
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    compression_ratio_hallucination_threshold=HALLUCINATION_COMPRESSION,
                    fp16=FP16,
                )
                print(f"{datetime.now()} - {file_name}: Chunk {i+1} produced {len(result['segments'])} segment(s).")
                for segment in result["segments"]:
                    segment['start'] += start_time
                    segment['end'] += start_time
                    segment['id'] = next_segment_id
                    next_segment_id += 1
                    for word in segment.get("words", []):
                        word['start'] += start_time
                        word['end'] += start_time
                    transcriptions.append(segment)

            all_transcriptions["Speaker 0"] = transcriptions
            print(f"{datetime.now()} - {file_name}: Completed transcription for mono file.")
            return file_name, all_transcriptions

        else:
            error_msg = f"{file_name}: Unsupported number of channels: {num_channels}"
            print(f"{datetime.now()} - {error_msg}")
            return file_name, {"error": error_msg}

    except Exception as e:
        error_msg = f"{file_name}: Error during transcription: {str(e)}"
        print(f"{datetime.now()} - {error_msg}")
        return file_name, {"error": error_msg}
    finally:
        # Optionally, clean up any resources specific to this process.
        pass

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        contents = await upload_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(contents)
            print(f"{datetime.now()} - Saved uploaded file {upload_file.filename} to temporary path {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe_audio_bulk")
async def transcribe_audio(
    languages: str = None,  # Comma-separated list (e.g., "auto,en,fr")
    files: List[UploadFile] = File(...),
):
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed.")

    # Split the languages string into a list if provided, otherwise default to "auto" for each file.
    if languages is None:
        languages_list = ["auto"] * len(files)
    else:
        languages_list = [lang.strip() for lang in languages.split(",")]
        if len(languages_list) != len(files):
            raise HTTPException(status_code=400, detail="The number of languages provided must match the number of files.")

    tmp_files = []
    filenames = []
    for file in files:
        tmp_file_path = await save_upload_file_tmp(file)
        tmp_files.append(tmp_file_path)
        filenames.append(file.filename)
    print(f"{datetime.now()} - Bulk endpoint: Received {len(files)} file(s) for transcription.")

    loop = asyncio.get_event_loop()
    tasks = []
    # Schedule each transcription task with its corresponding language.
    for tmp_file, filename, language in zip(tmp_files, filenames, languages_list):
        print(f"{datetime.now()} - Scheduling transcription for {filename} with language '{language}'")
        task = loop.run_in_executor(executor, transcribe_file_process, tmp_file, filename, language)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    output = {}
    for filename, transcription in results:
        output[filename] = transcription

    for tmp_file in tmp_files:
        os.remove(tmp_file)
        print(f"{datetime.now()} - Bulk endpoint: Removed temporary file {tmp_file}")

    return output

# ----- ADD: эндпоинт /transcribe_audio_batch_bulk (как /transcribe_audio_bulk, но с батчем) -----
@app.post("/transcribe_audio_local_batch")
async def transcribe_audio_local_batch(
    languages: str = None,   # Комма-сепаратед для локальных путей: "auto,ru,en"
    paths: List[str] = None, # Список локальных путей к файлам
):
    """
    Принимает список локальных файлов и транскрибирует их батчево (30s фреймы, BATCH_SIZE на GPU).
    Использует ту же логику и эвристики, что /transcribe_audio_batch_bulk.
    """
    if paths is None or len(paths) == 0:
        raise HTTPException(status_code=400, detail="Provide at least one local file path in 'paths'.")
    if len(paths) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed.")

    # Языки
    if languages is None:
        languages_list = ["auto"] * len(paths)
    else:
        languages_list = [lang.strip() for lang in languages.split(",")]
        if len(languages_list) != len(paths):
            raise HTTPException(status_code=400, detail="The number of languages must match the number of paths.")

    # Валидация путей
    for p in paths:
        if not os.path.isfile(p):
            raise HTTPException(status_code=400, detail=f"File does not exist: {p}")

    print(f"{datetime.now()} - LocalBatch: Transcribing {len(paths)} file(s) with batching")

    loop = asyncio.get_event_loop()
    tasks = []
    # РЕКОМЕНДАЦИЯ: для батч-режима лучше CONCURRENT_TRANSCRIPTIONS=1 (одна модель на один GPU)
    for path, language in zip(paths, languages_list):
        filename = os.path.basename(path)
        print(f"{datetime.now()} - LocalBatch: schedule {filename} (lang='{language}')")
        task = loop.run_in_executor(executor, transcribe_file_process_batched, path, filename, language)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    output = {}
    for filename, transcription in results:
        output[filename] = transcription

    print(f"{datetime.now()} - LocalBatch: Completed all files")

    #print(output)
    return output


@app.post("/transcribe_audio_local")
async def transcribe_audio_local(
    languages: str = None,  # Comma-separated list for local file paths
    paths: List[str] = None,
):
    """
    Accepts a list of local audio file paths and transcribes them 
    using the same chunk-based approach as /transcribe_audio_bulk.
    """
    if paths is None or len(paths) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed."
        )
    
    if languages is None:
        languages_list = ["auto"] * len(paths)
    else:
        languages_list = [lang.strip() for lang in languages.split(",")]
        if len(languages_list) != len(paths):
            raise HTTPException(status_code=400, detail="The number of languages provided must match the number of paths.")

    for path in paths:
        if not os.path.isfile(path):
            raise HTTPException(
                status_code=400,
                detail=f"File does not exist: {path}"
            )

    print(f"{datetime.now()} - Local endpoint: Transcribing {len(paths)} file(s) with provided languages.")
    loop = asyncio.get_event_loop()
    tasks = []
    for path, language in zip(paths, languages_list):
        filename = os.path.basename(path)
        print(f"{datetime.now()} - Local endpoint: Scheduling transcription for {filename} with language '{language}'")
        task = loop.run_in_executor(executor, transcribe_file_process, path, filename, language)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    output = {}
    for filename, transcription in results:
        output[filename] = transcription

    print(f"{datetime.now()} - Local endpoint: Completed transcription for all files.")
    return output

# Create a ProcessPoolExecutor with a limited number of processes, using the initializer
executor = ProcessPoolExecutor(
    max_workers=CONCURRENT_TRANSCRIPTIONS,
    initializer=worker_init
)
