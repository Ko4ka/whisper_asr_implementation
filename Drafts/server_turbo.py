import os
import torch
import asyncio
import concurrent.futures
import threading
import time
from functools import partial
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from tempfile import NamedTemporaryFile
import traceback
import warnings
import gc
import torchaudio


# Import pyannote libraries
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Import Transformers libraries
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()

# Some warnings are BS artifacts and can be ignored
warnings.filterwarnings(
    "ignore",
    message="Whisper did not predict an ending timestamp*",
    category=UserWarning
)

# IMPORTANT SETTINGS
HF_AUTH_TOKEN = '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'  # Replace with your actual Hugging Face token
DEFAULT_LANGUAGE = 'ru'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the new model using Transformers
model_id = "openai/whisper-large-v3-turbo"  # To download the required models
chunks_in_parallel = 8  # Adjust based on your system
max_workers = 10  # Affects I/O part; ML part is handled one-by-one due to thread safety
MAX_CONCURRENT_REQUESTS = 1  # Adjust based on your GPU capacity
CLEAR_DIARIZATION = False

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Create the transcription pipeline
transcription_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=60,  # Enable chunked processing
    batch_size=chunks_in_parallel,      # Adjust based on your system
    torch_dtype=torch_dtype,
    device=device
)

# Load the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_AUTH_TOKEN  # Replace with your Hugging Face token
)
diarization_pipeline.to(device)

# Semaphore to limit concurrency
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# Locks to ensure thread safety
transcription_lock = threading.Lock()
diarization_lock = threading.Lock()

# ThreadPoolExecutor for multithreading
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)  # Adjust based on your system

@app.post("/transcribe_mono")
async def transcribe_mono(
    audio_files: List[UploadFile] = File(...),
    diarization: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
    force_lang: Optional[str] = Form(None)
):
    if len(audio_files) > 10:
        raise HTTPException(status_code=400, detail="A maximum of 10 files is allowed.")

    # Attempt to acquire the semaphore
    acquired = request_semaphore.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server is busy processing other requests. Please try again later.")

    try:
        tasks = []
        lang = DEFAULT_LANGUAGE
        if force_lang:
            lang = force_lang
        for audio_file in audio_files:
            tasks.append(process_file(audio_file, diarization, num_speakers, lang))

        results = await asyncio.gather(*tasks)

        return JSONResponse(content={"results": results})

    finally:
        # Release the semaphore
        request_semaphore.release()

@app.post("/transcribe_stereo")
async def transcribe_stereo(
    audio_files: List[UploadFile] = File(...),
    force_lang: Optional[str] = Form(None)
):
    if len(audio_files) > 10:
        raise HTTPException(status_code=400, detail="A maximum of 10 files is allowed.")

    # Attempt to acquire the semaphore
    acquired = request_semaphore.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server is busy processing other requests. Please try again later.")

    try:
        tasks = []
        lang = DEFAULT_LANGUAGE
        if force_lang:
            lang = force_lang
        for audio_file in audio_files:
            tasks.append(process_file_stereo(audio_file, lang))

        results = await asyncio.gather(*tasks)

        return JSONResponse(content={"results": results})

    finally:
        # Release the semaphore
        request_semaphore.release()

async def process_file(audio_file, diarization, num_speakers, lang):
    # Save the uploaded file to a temporary location
    file_ext = audio_file.filename.split('.')[-1]
    with NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(await audio_file.read())

    try:
        # Offload CPU-bound transcription to a thread pool
        loop = asyncio.get_event_loop()
        start_time = time.time()
        transcription_future = loop.run_in_executor(
            thread_pool,
            partial(transcribe_audio, temp_file_name, language=lang)
        )

        # Perform diarization if requested
        if diarization:
            diarization_future = loop.run_in_executor(
                thread_pool,
                partial(diarize_audio, temp_file_name, num_speakers=num_speakers)
            )

            # Wait for both transcription and diarization to complete
            transcription_result, diarization_result = await asyncio.gather(
                transcription_future, diarization_future
            )

            transcription_text, transcription_segments, transcription_time = transcription_result
            diarization_time = diarization_result['diarization_time']
            # Update transcription time
            # print(f'{transcription_time} - {diarization_time}')
            diarization_result = diarization_result['diarization']

            combined_output = combine_transcription_and_diarization(
                transcription_segments,
                diarization_result
            )

            total_time = time.time() - start_time

            return {
                "file": audio_file.filename,
                "transcription": combined_output,
                "transcription_time": transcription_time - diarization_time,  # Diarization eats more time and timed after
                "diarization_time": diarization_time,
                "total_time": total_time
            }
        else:
            transcription_text, transcription_segments, transcription_time = await transcription_future
            total_time = time.time() - start_time

            return {
                "file": audio_file.filename,
                "transcription": transcription_text,
                "transcription_time": transcription_time,
                "total_time": total_time
            }

    except Exception as e:
        traceback.print_exc()
        return {"file": audio_file.filename, "error": str(e)}

    finally:
        # Clean up the temporary file
        os.remove(temp_file_name)

async def process_file_stereo(audio_file, lang):
    # Save the uploaded file to a temporary location
    file_ext = audio_file.filename.split('.')[-1]
    with NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(await audio_file.read())

    temp_files_to_remove = [temp_file_name]
    try:
        # Load audio data using torchaudio
        import torchaudio
        waveform, sample_rate = torchaudio.load(temp_file_name)

        # Check if audio is stereo
        if waveform.shape[0] != 2:
            raise HTTPException(status_code=400, detail=f"File {audio_file.filename} is not a stereo audio file.")

        # Extract channels
        channel_0_waveform = waveform[0, :].unsqueeze(0)  # [1, num_samples]
        channel_1_waveform = waveform[1, :].unsqueeze(0)  # [1, num_samples]

        # Save each channel to a temporary file
        with NamedTemporaryFile(delete=False, suffix=f"_ch0.{file_ext}") as temp_ch0_file:
            temp_ch0_file_name = temp_ch0_file.name
            temp_files_to_remove.append(temp_ch0_file_name)
            torchaudio.save(temp_ch0_file_name, channel_0_waveform, sample_rate)

        with NamedTemporaryFile(delete=False, suffix=f"_ch1.{file_ext}") as temp_ch1_file:
            temp_ch1_file_name = temp_ch1_file.name
            temp_files_to_remove.append(temp_ch1_file_name)
            torchaudio.save(temp_ch1_file_name, channel_1_waveform, sample_rate)

        # Offload CPU-bound transcription to a thread pool
        loop = asyncio.get_event_loop()
        start_time = time.time()
        transcription_future_ch0 = loop.run_in_executor(
            thread_pool,
            partial(transcribe_audio, temp_ch0_file_name, language=lang)
        )
        transcription_future_ch1 = loop.run_in_executor(
            thread_pool,
            partial(transcribe_audio, temp_ch1_file_name, language=lang)
        )

        # Wait for both transcriptions to complete
        transcription_result_ch0, transcription_result_ch1 = await asyncio.gather(
            transcription_future_ch0, transcription_future_ch1
        )

        _, transcription_segments_ch0, transcription_time_ch0 = transcription_result_ch0
        _, transcription_segments_ch1, transcription_time_ch1 = transcription_result_ch1

        total_time = time.time() - start_time

        # Add speaker info to segments
        for segment in transcription_segments_ch0:
            segment['speaker'] = 'SPEAKER_0'
        for segment in transcription_segments_ch1:
            segment['speaker'] = 'SPEAKER_1'

        # Combine segments and sort by start time
        combined_segments = transcription_segments_ch0 + transcription_segments_ch1
        combined_segments.sort(key=lambda x: x['start'])

        return {
            "file": audio_file.filename,
            "segments": combined_segments,
            "total_time": total_time
        }

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions

    except Exception as e:
        traceback.print_exc()
        return {"file": audio_file.filename, "error": str(e)}

    finally:
        # Clean up the temporary files
        for temp_file in temp_files_to_remove:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def transcribe_audio(file_path, language="ru"):
    """
    Transcribe an audio file using the Transformers pipeline.
    """
    global transcription_pipeline
    if transcription_pipeline is None:
        raise Exception("Transcription pipeline is not loaded.")

    transcription_start_time = time.time()

    with transcription_lock:
        try:
            # Set generate_kwargs for language and other options
            generate_kwargs = {"language": language}
            result = transcription_pipeline(
                file_path,
                generate_kwargs=generate_kwargs,
                return_timestamps=True
            )
        except Exception as e:
            print('Transcription Error')
            traceback.print_exc()
            raise Exception(f"Error transcribing audio file: {str(e)}")

        # Collect the transcription text and segments
        full_transcription = result['text']
        all_segments = []
        for chunk in result.get('chunks', []):
            segment = {
                'start': chunk['timestamp'][0],
                'end': chunk['timestamp'][1],
                'text': chunk['text']
            }
            all_segments.append(segment)

    transcription_end_time = time.time()
    transcription_time = transcription_end_time - transcription_start_time

    return full_transcription.strip(), all_segments, transcription_time

def diarize_audio(file_path, num_speakers=None):
    """
    Perform speaker diarization on the audio file.
    """
    global diarization_pipeline
    if diarization_pipeline is None:
        raise Exception("Diarization pipeline is not loaded.")

    diarization_start_time = time.time()

    with diarization_lock:
        try:
            # Load audio data using torchaudio
            waveform, sample_rate = torchaudio.load(file_path)

            # Prepare the input for the pipeline
            diarization_input = {
                'waveform': waveform,
                'sample_rate': sample_rate
            }
            if num_speakers:
                diarization = diarization_pipeline(diarization_input, num_speakers=num_speakers)
            else:
                diarization = diarization_pipeline(diarization_input)
        except Exception as e:
            print('Diarization Error')
            traceback.print_exc()
            raise Exception(f"Diarization error: {str(e)}")

    diarization_end_time = time.time()
    diarization_time = diarization_end_time - diarization_start_time
    
    if CLEAR_DIARIZATION:
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

    return {
        'diarization': diarization,
        'diarization_time': diarization_time
    }

def combine_transcription_and_diarization(transcription_segments, diarization):
    """
    Combine transcription segments with diarization results.
    """
    combined_output = []

    for segment in transcription_segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']

        # Define the segment interval
        segment_interval = Segment(start, end)
        
        # Get speaker labels within the segment interval
        speaker_labels = diarization.crop(segment_interval)

        # Calculate speaking time for each speaker within this segment
        speaker_durations = {}
        for time_interval, _, speaker in speaker_labels.itertracks(yield_label=True):
            start_time = time_interval.start
            end_time = time_interval.end
            duration = end_time - start_time

            if speaker in speaker_durations:
                speaker_durations[speaker] += duration
            else:
                speaker_durations[speaker] = duration

        # Assign speaker with the longest duration within this segment
        if speaker_durations:
            speaker = max(speaker_durations, key=speaker_durations.get)
        else:
            speaker = "Unknown"

        combined_output.append({
            'start': start,
            'end': end,
            'text': text,
            'speaker': speaker,
            'diar_classes': str(speaker_labels)
        })

    return combined_output



