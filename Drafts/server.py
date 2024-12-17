import os
import torch
import whisper
import concurrent.futures
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from typing import Optional
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Segment

app = FastAPI()

TOKEN = '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'

# Load models at startup to avoid reloading on each request
whisper_model = whisper.load_model("large")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=TOKEN  # Replace with your Hugging Face token
)
diarization_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

@app.post("/transcribe_mono")
async def transcribe(
    audio_file: UploadFile = File(...),
    diarization: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
    force_lang: Optional[str] = Form(None)
):
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(await audio_file.read())

    try:
        # Transcribe the audio
        lang = 'ru'
        if force_lang:
            lang = force_lang
        transcription_text, transcription_segments = transcribe_audio(
            temp_file_name,
            language=lang
        )

        # Perform diarization if requested
        if diarization:
            diarization_result = diarize_audio(
                temp_file_name,
                num_speakers=num_speakers
            )
            combined_output = combine_transcription_and_diarization(
                transcription_segments,
                diarization_result
            )
            return JSONResponse(content={"transcription": combined_output})
        else:
            return JSONResponse(content={"transcription": transcription_text})

    finally:
        # Clean up the temporary file
        os.remove(temp_file_name)

def transcribe_audio(file_path, language="ru", segment_duration=30):
    """
    Transcribe an audio file in chunks of specified duration.
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

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

    return full_transcription.strip(), all_segments

def diarize_audio(file_path, num_speakers=None):
    """
    Perform speaker diarization on the audio file.
    """
    if num_speakers:
        diarization = diarization_pipeline(file_path, num_speakers=num_speakers)
    else:
        diarization = diarization_pipeline(file_path)
    return diarization

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
