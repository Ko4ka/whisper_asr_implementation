from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pyannote.audio import Pipeline
from pyannote.audio.utils.reproducibility import ReproducibilityWarning


import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import tempfile
import os
from typing import List
from datetime import datetime
import traceback
import torchaudio
from typing import Optional
import warnings

# Misleading warnings
warnings.filterwarnings("ignore", category=ReproducibilityWarning)

# Config
MAX_FILES_PER_REQUEST = 25  # Maximum number of files per request
CONCURRENT_PROCESSES = 2  # Number of concurrent diarizations - about 1.8 GB per stream

# Diarization Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token='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'  # Replace with your Hugging Face token
)
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

app = FastAPI()

def diarize_file_process(file_path: str, file_name: str, num_speakers):
    """
    Process an audio file for speaker diarization.
    """
    print(f'{datetime.now()} Diarizing {file_name} Num speakers {num_speakers}')
    try:
         # Check audio properties
        waveform, sample_rate = torchaudio.load(file_path)
        num_channels = waveform.shape[0]

        if num_channels != 1:
            return file_name, {"error": "Error: Audio is not mono. Please provide a mono audio file."}

        if num_speakers:
            diarization = pipeline(file_path, num_speakers= int(num_speakers))
        else:
            diarization = pipeline(file_path)

        # Format diarization results
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        return file_name, {"diarization": results}
    except Exception as e:
        return file_name, {"error": str(e)}

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Save uploaded file to a temporary location.
    """
    try:
        contents = await upload_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(contents)
            return tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diarize_audio_bulk")
async def diarize_audio(files: List[UploadFile] = File(...),
                        num_speakers: Optional[int] = Form(None)):
    """
    API endpoint to diarize multiple audio files concurrently.
    """
    try:
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
            # Schedule diarize_file_process to run in ProcessPoolExecutor
            task = loop.run_in_executor(executor, diarize_file_process, tmp_file, filename, num_speakers)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Map filenames to results
        output = {}
        for filename, diarization in results:
            output[filename] = diarization

        # Clean up temporary files
        for tmp_file in tmp_files:
            os.remove(tmp_file)

        return output
    except Exception as e:
        traceback.print_exc()
        return e


# Create a ProcessPoolExecutor with a limited number of processes
executor = ProcessPoolExecutor(max_workers=CONCURRENT_PROCESSES)
