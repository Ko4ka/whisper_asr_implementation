import os
import torch
import whisper
import asyncio
import concurrent.futures
import threading
import time
from functools import partial
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Segment
from tempfile import NamedTemporaryFile
import traceback

app = FastAPI()

# Load models at startup (global instances)
whisper_model = whisper.load_model("large-v3-turbo")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token='YOUR_HF_AUTH_TOKEN'  # Replace with your Hugging Face token
)
diarization_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Locks to ensure thread safety
whisper_lock = threading.Lock()
diarization_lock = threading.Lock()

# ThreadPoolExecutor for multithreading
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Adjust based on your system

@app.post("/transcribe_mono")
async def transcribe(
    audio_files: List[UploadFile] = File(...),
    diarization: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
    force_lang: Optional[str] = Form(None)
):
    if len(audio_files) > 10:
        raise HTTPException(status_code=400, detail="A maximum of 10 files is allowed.")

    tasks = []
    lang = 'ru'
    if force_lang:
        lang = force_lang
    for audio_file in audio_files:
        tasks.append(process_file(audio_file, diarization, num_speakers, lang))

    results = await asyncio.gather(*tasks)

    return JSONResponse(content={"results": results})

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
            diarization_result = diarization_result['diarization']

            combined_output = combine_transcription_and_diarization(
                transcription_segments,
                diarization_result
            )

            total_time = time.time() - start_time

            return {
                "file": audio_file.filename,
                "transcription": combined_output,
                "transcription_time": transcription_time,
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

def transcribe_audio(file_path, language="ru", segment_duration=30):
    """
    Transcribe an audio file in chunks of specified duration.
    """
    global whisper_model
    if whisper_model is None:
        raise Exception("Whisper model is not loaded.")

    transcription_start_time = time.time()

    with whisper_lock:
        # Load the audio file
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            print('Transcription Error')
            traceback.print_exc()
            raise Exception(f"Error loading audio file: {str(e)}")

        # Segment duration in milliseconds
        segment_duration_ms = segment_duration * 60 * 1000
        audio_length_ms = len(audio)

        # Store full transcription and all segments
        full_transcription = ""
        all_segments = []

        # Process the audio in chunks
        for i in range(0, audio_length_ms, segment_duration_ms):
            # Extract a chunk of audio
            chunk = audio[i:i+segment_duration_ms]

            # Export the chunk to a temporary file
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                chunk.export(temp_chunk_file.name, format="wav")
                temp_chunk_file_name = temp_chunk_file.name

            # Transcribe the chunk
            result = whisper_model.transcribe(temp_chunk_file_name, language=language)

            # Collect the transcription text and segments
            full_transcription += result['text'] + " "
            for segment in result['segments']:
                # Adjust segment timings based on the chunk's start time
                segment['start'] += i / 1000
                segment['end'] += i / 1000
                all_segments.append(segment)

            # Remove the temporary file after transcription
            os.remove(temp_chunk_file_name)

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
            if num_speakers:
                diarization = diarization_pipeline({"audio": file_path}, num_speakers=num_speakers)
            else:
                diarization = diarization_pipeline({"audio": file_path})
        except Exception as e:
            print('Diarization Error')
            traceback.print_exc()
            raise Exception(f"Diarization error: {str(e)}")

    diarization_end_time = time.time()
    diarization_time = diarization_end_time - diarization_start_time

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

        # Find which speaker was speaking during this segment
        segment_interval = Segment(start, end)
        speaker_labels = diarization.crop(segment_interval)

        # Assume the most frequent speaker label is the correct one
        speakers = [label for _, _, label in speaker_labels.itertracks(yield_label=True)]
        if speakers:
            speaker = max(set(speakers), key=speakers.count)
        else:
            speaker = "Unknown"

        combined_output.append({
            'start': start,
            'end': end,
            'text': text,
            'speaker': speaker
        })

    return combined_output
