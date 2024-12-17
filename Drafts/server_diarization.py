import os
import torch
import asyncio
import threading
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from tempfile import NamedTemporaryFile
import traceback
import warnings
import torchaudio
import uvicorn
import json

# Import pyannote libraries
from pyannote.audio import Pipeline
from pyannote.core import Segment

app = FastAPI()

# Ignore specific warnings if needed
warnings.filterwarnings(
    "ignore",
    message="Some warning message*",
    category=UserWarning
)

# IMPORTANT SETTINGS
HF_AUTH_TOKEN = 'YOUR_HF_AUTH_TOKEN'  # Replace with your actual Hugging Face token

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "",
    use_auth_token=HF_AUTH_TOKEN  # Replace with your Hugging Face token
)
diarization_pipeline.to(device)

# Semaphore to limit concurrency (adjust as needed)
MAX_CONCURRENT_REQUESTS = 1
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# Lock to ensure thread safety
diarization_lock = threading.Lock()

@app.post("/diarize")
async def diarize(
    audio_files: List[UploadFile] = File(...),
    transcription_segments_list: List[str] = Form(...),
    num_speakers: Optional[int] = Form(None)
):
    """
    Endpoint to perform speaker diarization on multiple audio files using provided transcription segments.
    """
    if len(audio_files) > 10:
        raise HTTPException(status_code=400, detail="A maximum of 10 files is allowed.")

    # Attempt to acquire the semaphore
    acquired = request_semaphore.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing other requests. Please try again later."
        )

    try:
        # Ensure the number of audio files matches the number of transcription segments
        if len(audio_files) != len(transcription_segments_list):
            raise HTTPException(
                status_code=400,
                detail="The number of audio files must match the number of transcription segments."
            )

        tasks = []
        for audio_file, transcription_segments_str in zip(audio_files, transcription_segments_list):
            tasks.append(process_file(audio_file, transcription_segments_str, num_speakers))

        results = await asyncio.gather(*tasks)

        return JSONResponse(content={"results": results})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # Release the semaphore
        request_semaphore.release()

async def process_file(audio_file, transcription_segments_str, num_speakers):
    # Parse transcription segments from JSON string
    try:
        transcription_segments = json.loads(transcription_segments_str)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid format for transcription_segments. Expected JSON string."
        )

    # Save the uploaded file to a temporary location
    file_ext = audio_file.filename.split('.')[-1]
    with NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_audio_file:
        temp_audio_file_name = temp_audio_file.name
        temp_audio_file.write(await audio_file.read())

    try:
        start_time = time.time()

        # Perform diarization
        diarization_result, diarization_time = await perform_diarization(
            temp_audio_file_name,
            num_speakers
        )

        # Combine transcription segments with diarization
        combined_segments = combine_transcription_and_diarization(
            transcription_segments,
            diarization_result
        )

        total_time = time.time() - start_time

        return {
            "file": audio_file.filename,
            "diarized_segments": combined_segments,
            "diarization_time": diarization_time,
            "total_time": total_time
        }

    except Exception as e:
        traceback.print_exc()
        return {"file": audio_file.filename, "error": str(e)}

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_file_name):
            os.remove(temp_audio_file_name)

async def perform_diarization(file_path, num_speakers=None):
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

    return diarization, diarization_time

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
