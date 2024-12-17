from fastapi import FastAPI, UploadFile, File, HTTPException
import asyncio
import whisper
import torchaudio
import numpy as np
import torch
import tempfile
import os
import warnings


# Config
CHUNK_DURATION = 180
IGNORE_WIN_WARNINGS = True

app = FastAPI()

# Load the Whisper model on GPU once during startup
model = whisper.load_model("turbo", device="cuda")

# Create an asyncio.Lock to prevent concurrent transcriptions
transcription_lock = asyncio.Lock()

# Triton and flashAttention warnings
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


@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    # Check if the transcription service is busy
    if transcription_lock.locked():
        raise HTTPException(status_code=503, detail="Service busy")

    # Acquire the lock to start transcription
    await transcription_lock.acquire()
    try:
        # Read the uploaded file
        contents = await file.read()

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(dir='./temp', delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(tmp_file_path)

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
                        result = model.transcribe(
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

                        # Adjust segment times
                        for segment in result["segments"]:
                            segment['start'] += start_time
                            segment['end'] += start_time
                            transcriptions.append(segment)

                    # Collect transcriptions for the current speaker
                    all_transcriptions[channel_name] = transcriptions

                # Return the transcriptions as JSON
                return all_transcriptions

            else:
                return {"error": "Error: Audio is not stereo."}

        finally:
            # Clean up the temporary file
            os.remove(tmp_file_path)

    finally:
        # Release the lock after transcription is complete
        transcription_lock.release()
