from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import asyncio
import whisper
import torchaudio
import numpy as np
import torch
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

def worker_init():
    """
    Function to initialize the model in each worker process.
    This function will be called when each process starts.
    """
    global model
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
