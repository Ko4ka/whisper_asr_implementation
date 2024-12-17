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
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import ffmpeg

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
    )  # Когда-нибудь это говно заработает на винде... но только не сегодня

    warnings.filterwarnings(
        "ignore",
        message=".*Failed to launch Triton kernels.*",
        category=UserWarning,
        module=".*whisper\.timing"
    )  # А это и завтра не заработает...

# Global variable to hold the model in worker processes
model = None

def worker_init():
    """
    Function to initialize the model in each worker process.
    This function will be called when each process starts.
    """
    global model
    # Load the model once when the process starts
    model = whisper.load_model("turbo", device="cuda")
    print(f"Model loaded in process {os.getpid()}")

def transcribe_file_process(file_path, file_name):
    print(f'{datetime.now()} Transcribing {file_name}')
    global model
    try:
        if model is None:
            # This should not happen if the model is loaded during initialization
            model = whisper.load_model("turbo", device="cuda")

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
        all_transcriptions = {}

        # Ensure the audio is stereo
        if num_channels == 2:

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
                    result = model.transcribe(
                        audio=chunk_numpy,
                        language="ru",
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

                    # Adjust segment times
                    for segment in result["segments"]:
                        segment['start'] += start_time
                        segment['end'] += start_time
                        transcriptions.append(segment)

                # Collect transcriptions for the current speaker
                all_transcriptions[channel_name] = transcriptions

            # Return the transcriptions as JSON
            return file_name, all_transcriptions
        
        elif num_channels == 1:  # Mono file
            # Process the single channel
            transcriptions = []
            num_samples = waveform.shape[1]
            num_chunks = (num_samples + chunk_samples - 1) // chunk_samples

            # IDs to adjust
            next_segment_id = 0  # To ensure unique segment IDs across chunks

            for i in range(num_chunks):
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, num_samples)
                chunk_waveform = waveform[:, start_sample:end_sample]
                start_time = start_sample / sample_rate

                # Convert to NumPy array
                chunk_numpy = chunk_waveform.squeeze().numpy()

                # Transcribe the chunk
                result = model.transcribe(
                    audio=chunk_numpy,
                    language="ru",
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

                '''# Adjust segment times
                for segment in result["segments"]:
                    segment['start'] += start_time
                    segment['end'] += start_time
                    transcriptions.append(segment)'''

                # Adjust segment times
                for segment in result["segments"]:
                    # Adjust segment-level start and end times
                    segment['start'] += start_time
                    segment['end'] += start_time

                    # Assign a unique segment ID
                    segment['id'] = next_segment_id
                    next_segment_id += 1

                    # Adjust word-level timings
                    for word in segment.get("words", []):
                        word['start'] += start_time
                        word['end'] += start_time

                    transcriptions.append(segment)

            all_transcriptions["Speaker 0"] = transcriptions
             # Return the transcriptions as JSON
            return file_name, all_transcriptions

        else:
            return file_name, {"error": "Error: Audio has unsupported number of channels."}

    except Exception as e:
        return file_name, {"error": str(e)}
    finally:
        # Clean up resources if necessary
        pass  # No action needed here

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

    loop = asyncio.get_event_loop()
    tasks = []
    for tmp_file, filename in zip(tmp_files, filenames):
        # Schedule the transcribe_file_process function to run in the ProcessPoolExecutor
        task = loop.run_in_executor(executor, transcribe_file_process, tmp_file, filename)
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

# Create a ProcessPoolExecutor with a limited number of processes, using the initializer
executor = ProcessPoolExecutor(
    max_workers=CONCURRENT_TRANSCRIPTIONS,
    initializer=worker_init
)
