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
CONCURRENT_PROCESSES = 1  # Number of concurrent diarizations - about 1.8 GB per stream

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
                        num_speakers: Optional[List[Optional[int]]] = Form(None)):
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
        # Iterate over files and pass the corresponding num_speakers (if provided)
        for idx, (tmp_file, filename) in enumerate(zip(tmp_files, filenames)):
            current_num_speakers = None
            if num_speakers is not None and len(num_speakers) > idx:
                current_num_speakers = num_speakers[idx]
            task = loop.run_in_executor(
                executor, diarize_file_process, tmp_file, filename, current_num_speakers
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Map filenames to results
        output = {filename: result for filename, result in results}

        # Clean up temporary files
        for tmp_file in tmp_files:
            os.remove(tmp_file)

        return output
    except Exception as e:
        traceback.print_exc()
        return e
    
@app.post("/diarize_audio_bulk_local")
async def diarize_audio_bulk_local(
    paths: List[str], 
    num_speakers: Optional[int] = None  # Note: if you want per-file values, consider using Optional[List[int]]
):
    """
    Accepts a list of local audio file paths and optionally num_speakers.
    Diarizes the audio using the same pipeline as /diarize_audio_bulk.
    """
    print("Received paths:", paths)
    print("Received num_speakers:", num_speakers)
    
    # 1. Validate number of paths
    if len(paths) > MAX_FILES_PER_REQUEST:
        err_msg = f"Maximum of {MAX_FILES_PER_REQUEST} files allowed."
        print(err_msg)
        raise HTTPException(status_code=400, detail=err_msg)

    # 2. Validate that each path exists
    for path in paths:
        print("Validating path:", path)
        if not os.path.isfile(path):
            err_msg = f"File does not exist or is not accessible: {path}"
            print(err_msg)
            raise HTTPException(status_code=400, detail=err_msg)
    
    # 3. Create concurrency tasks
    loop = asyncio.get_event_loop()
    tasks = []
    for idx, path in enumerate(paths):
        filename = os.path.basename(path)
        # If num_speakers is provided as a single int, use it for all files.
        # If you plan to provide a list of speaker counts per file, adjust the type accordingly.
        current_num_speakers = num_speakers
        print(f"Creating task for file {idx}:")
        print(f"  Path: {path}")
        print(f"  Filename: {filename}")
        print(f"  num_speakers: {current_num_speakers}")
        task = loop.run_in_executor(
            executor, 
            diarize_file_process, 
            path, 
            filename, 
            current_num_speakers
        )
        tasks.append(task)
    
    # 4. Run all tasks concurrently
    print("Running diarization tasks concurrently...")
    results = await asyncio.gather(*tasks)
    print("Diarization tasks completed. Raw results:", results)
    
    # 5. Build output as { filename: <diarization or error> }
    output = {filename: result for filename, result in results}
    print("Returning output:", output)
    return output


# Create a ProcessPoolExecutor with a limited number of processes
executor = ProcessPoolExecutor(max_workers=CONCURRENT_PROCESSES)
