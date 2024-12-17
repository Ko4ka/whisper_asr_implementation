from fastapi import FastAPI, UploadFile, File, HTTPException
import asyncio
import whisper
import torchaudio
import numpy as np
import torch
import tempfile
import os
import warnings
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Config
CHUNK_DURATION = 180
IGNORE_WIN_WARNINGS = True
MAX_FILES_PER_REQUEST = 10  # Maximum number of files per request
CONCURRENT_TRANSCRIPTIONS = 4  # Maximum number of concurrent transcriptions

app = FastAPI()

# Suppress specific warnings if needed
if IGNORE_WIN_WARNINGS:
    # Suppress the 'Torch was not compiled with flash attention' warning
    warnings.filterwarnings(
        "ignore",
        message=".*Torch was not compiled with flash attention.*",
        category=UserWarning,
        module=".*whisper\.model"
    )

    # Suppress the 'Failed to launch Triton kernels' warnings from whisper.timing
    warnings.filterwarnings(
        "ignore",
        message=".*Failed to launch Triton kernels.*",
        category=UserWarning,
        module=".*whisper\.timing"
    )

# Pre-load the required number of model instances
model_instances = []

def load_model_instances():
    global model_instances
    for _ in range(CONCURRENT_TRANSCRIPTIONS):
        model = whisper.load_model("turbo", device="cuda")
        model_instances.append(model)
        print(f'Loaded instance {_+1}')

load_model_instances()

# Create a semaphore to limit access to model instances
model_semaphore = asyncio.Semaphore(CONCURRENT_TRANSCRIPTIONS)

async def transcribe_file(file_path, file_name):
    await model_semaphore.acquire()
    try:
        # Get a model from the pool
        model = model_instances.pop()

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample to 16000 Hz if necessary
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
            sample_rate = target_sample_rate

        # Parameters
        chunk_duration = CHUNK_DURATION  # Chunk duration in seconds
        num_channels = waveform.shape[0]
        chunk_samples = int(chunk_duration * sample_rate)

        # Ensure the audio is stereo
        if num_channels == 2:
            all_transcriptions = {}

            for channel_idx in range(num_channels):
                # Process each channel separately
                channel_waveform = waveform[channel_idx].unsqueeze(0)
                channel_name = f"Speaker {channel_idx}"

                # Split into chunks
                num_samples = channel_waveform.shape[1]
                num_chunks = (num_samples + chunk_samples - 1) // chunk_samples
                transcriptions = []

                for i in range(num_chunks):
                    start_sample = i * chunk_samples
                    end_sample = min((i + 1) * chunk_samples, num_samples)
                    chunk_waveform = channel_waveform[:, start_sample:end_sample]
                    start_time = start_sample / sample_rate

                    # Convert to NumPy array
                    chunk_numpy = chunk_waveform.squeeze().numpy()

                    # Transcribe the chunk
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: model.transcribe(
                            audio=chunk_numpy,
                            language="ru",
                            temperature=(0.0, 0.1),
                            no_speech_threshold=0.3,
                            suppress_tokens=[
                                50365, 2933, 8893, 403, 1635, 10461, 40653,
                                413, 4775, 51, 284, 89, 453, 51864, 50366,
                                8567, 1435, 21403, 5627, 15363, 17781, 485,
                                51863
                            ],
                            condition_on_previous_text=False,
                            word_timestamps=True,
                            compression_ratio_hallucination_threshold=2.1,
                            fp16=True,
                        )
                    )

                    # Adjust segment times
                    for segment in result["segments"]:
                        segment['start'] += start_time
                        segment['end'] += start_time
                        transcriptions.append(segment)

                # Collect transcriptions for the current speaker
                all_transcriptions[channel_name] = transcriptions

            # Return the transcriptions as JSON
            return file_name, all_transcriptions

        else:
            return file_name, {"error": "Error: Audio is not stereo."}

    except Exception as e:
        return file_name, {"error": str(e)}
    finally:
        # Put the model back into the pool and release the semaphore
        model_instances.append(model)
        model_semaphore.release()

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        contents = await upload_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(contents)
            return tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe_audio_bulk")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed.")

    # Save uploaded files to temporary files
    tmp_files = []
    filenames = []
    for file in files:
        tmp_file_path = await save_upload_file_tmp(file)
        tmp_files.append(tmp_file_path)
        filenames.append(file.filename)

    tasks = []
    for tmp_file, filename in zip(tmp_files, filenames):
        # Schedule the transcribe_file coroutine
        task = transcribe_file(tmp_file, filename)
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Map filenames to results
    output = {}
    for filename, transcription in results:
        output[filename] = transcription

    # Clean up temporary files
    for tmp_file in tmp_files:
        os.remove(tmp_file)

    return output