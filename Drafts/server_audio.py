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
import torchaudio

# Import Transformers libraries
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()

# Ignore specific warnings
warnings.filterwarnings(
    "ignore",
    message="Whisper did not predict an ending timestamp*",
    category=UserWarning
)

# IMPORTANT SETTINGS
HF_AUTH_TOKEN = 'YOUR_HF_AUTH_TOKEN'  # Replace with your actual Hugging Face token
DEFAULT_LANGUAGE = 'ru'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model using Transformers
model_id = "openai/whisper-large-v3-turbo"
chunks_in_parallel = 16  # Adjust based on your system
max_workers = 10  # Adjust based on your system
MAX_CONCURRENT_REQUESTS = 1  # Adjust based on your GPU capacity

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
    chunk_length_s=30,  # Adjust based on your needs
    batch_size=chunks_in_parallel,  # Adjust based on your system
    torch_dtype=torch_dtype,
    device=device
)

# Semaphore to limit concurrency
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# Lock to ensure thread safety
transcription_lock = threading.Lock()

# ThreadPoolExecutor for multithreading
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

@app.post("/transcribe_mono")
async def transcribe_mono(
    audio_files: List[UploadFile] = File(...),
    force_lang: Optional[str] = Form(None)
):
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
        tasks = []
        lang = DEFAULT_LANGUAGE
        if force_lang:
            lang = force_lang
        for audio_file in audio_files:
            tasks.append(process_file(audio_file, lang))

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
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing other requests. Please try again later."
        )

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

async def process_file(audio_file, lang):
    # Save the uploaded file to a temporary location
    file_ext = audio_file.filename.split('.')[-1]
    with NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(await audio_file.read())

    try:
        # Offload transcription to a thread pool
        loop = asyncio.get_event_loop()
        start_time = time.time()
        transcription_future = loop.run_in_executor(
            thread_pool,
            partial(transcribe_audio, temp_file_name, language=lang)
        )

        transcription_text, transcription_segments, transcription_time = await transcription_future
        total_time = time.time() - start_time

        return {
            "file": audio_file.filename,
            "segments": transcription_segments,
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
        waveform, sample_rate = torchaudio.load(temp_file_name)

        # Check if audio is stereo
        if waveform.shape[0] != 2:
            raise HTTPException(
                status_code=400,
                detail=f"File {audio_file.filename} is not a stereo audio file."
            )

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

        # Offload transcription to a thread pool
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

        _, transcription_segments_ch0, _ = transcription_result_ch0
        _, transcription_segments_ch1, _ = transcription_result_ch1

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
