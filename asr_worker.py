#!/usr/bin/env python3

import os
import time
import json
import sqlite3
import asyncio
import datetime
import tempfile
import subprocess
import traceback
from typing import List, Dict, Any, Union
import httpx

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS & CONFIG
# ---------------------------------------------------------------------------
DATABASE = "./Drafts/asr_queue.db"        # Path to your SQLite database
ASR_URL = "http://127.0.0.1:8000/transcribe_audio_local"
DIAR_URL = "http://127.0.0.1:8001/diarize_audio_bulk_local"
PROCESS_URL = "http://127.0.0.1:8002/process-transcription"

# Timeouts
ASR_TIMEOUT = 1080.0   # Timeout in seconds for ASR requests
DIAR_TIMEOUT = 380.0   # Timeout in seconds for diarization requests

# Audio post-processing constants
MAX_SPEECH_BUBBLE = 20.0
PAUSE_THRESHOLD = 2.0

# ASR settings
DEFAULT_LANGUAGE = None

# ---------------------------------------------------------------------------
# DATABASE-RELATED FUNCTIONS
# ---------------------------------------------------------------------------

def get_in_queue_jobs(db_path=DATABASE) -> List[Dict[str, Any]]:
    """
    Fetch all rows with status='in queue' from the jobs table and parse JSON fields.
    Returns a list of job dicts.
    """
    print("[get_in_queue_jobs] Checking DB for 'in queue' jobs.")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    columns = [
        "job_id",
        "added_at",
        "source",
        "source_data",
        "files",
        "size",
        "status",
        "transcription",
        "transcribed_at",
        "time_taken",
        "length"
    ]
    try:
        cursor.execute("SELECT * FROM jobs WHERE status = 'in queue'")
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"[get_in_queue_jobs] DB error: {e}")
        rows = []
    finally:
        conn.close()

    results = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        # Parse 'source_data' JSON
        if row_dict["source_data"]:
            try:
                row_dict["source_data"] = json.loads(row_dict["source_data"])
            except json.JSONDecodeError:
                row_dict["source_data"] = None

        # Parse 'files' JSON (often a list, but user might store string)
        if row_dict["files"]:
            try:
                row_dict["files"] = json.loads(row_dict["files"])
            except json.JSONDecodeError:
                row_dict["files"] = []

        results.append(row_dict)
    print(f"[get_in_queue_jobs] Found {len(results)} job(s) in queue.")
    return results

def update_job_in_db(job: dict, db_path=DATABASE) -> None:
    """
    Update a job in the 'jobs' table with the new status/transcription/time/etc.
    """
    print(f"[update_job_in_db] Updating job_id={job['job_id']} in DB with new data.")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Convert transcription to JSON string if it's a list or dict
    transcription_data = job.get("transcription")
    if isinstance(transcription_data, (dict, list)):
        transcription_data = json.dumps(transcription_data, ensure_ascii=False)
    else:
        transcription_data = str(transcription_data) if transcription_data else None

    sql = """
    UPDATE jobs
    SET
        size = ?,
        status = ?,
        transcription = ?,
        transcribed_at = ?,
        time_taken = ?,
        length = ?
    WHERE job_id = ?
    """
    cursor.execute(
        sql,
        (
            job.get("size"),            
            job.get("status"),          
            transcription_data,         
            job.get("transcribed_at"),  
            job.get("time_taken"),
            job.get("length"),      
            job["job_id"],
        )
    )
    conn.commit()
    conn.close()
    print(f"[update_job_in_db] Finished updating job_id={job['job_id']}.")

# ---------------------------------------------------------------------------
# AUDIO / ASR UTILITIES
# ---------------------------------------------------------------------------

def convert_to_mono_wav(input_path: str, output_path: str) -> None:
    """
    Converts an audio/video file to a mono WAV file with 16kHz sample rate using FFmpeg.
    Raises an exception if conversion fails.
    """
    try:
        if not output_path.lower().endswith('.wav'):
            raise ValueError("Output file must have a .wav extension")
        command = [
            "ffmpeg",
            "-y",                
            "-i", input_path,    
            "-ac", "1",          
            "-acodec", "pcm_s16le",  
            "-ar", "16000",      
            output_path
        ]
        subprocess.run(command, check=True)
        if not os.path.exists(output_path):
            raise RuntimeError("Conversion failed; output file not created.")
        
        # ─── get duration with ffprobe ───
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            output_path
        ]
        res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        duration_secs = float(res.stdout.strip())
        return int(duration_secs)
        # ───────────────────────────────────

    except Exception as e:
        print("[convert_to_mono_wav] Error:", e)
        traceback.print_exc()
        raise

def ensure_list(file_entry: Union[str, List[str]]) -> List[str]:
    """
    If 'file_entry' is a single string, wrap it in a list. If it's already a list, return as-is.
    """
    return file_entry if isinstance(file_entry, list) else [file_entry]

def create_speech_bubbles_mono(
    asr_transcription: Dict[str, List[Dict[str, Any]]],
    pause_threshold: float,
    max_duration: float
) -> List[Dict[str, Any]]:
    """
    Convert a mono (ASR-only) transcription dictionary into a list of speech bubbles.
    Consecutive segments for the same speaker are merged if gap < pause_threshold 
    and duration < max_duration.
    """
    bubbles = []
    for speaker, segments in asr_transcription.items():
        segments.sort(key=lambda seg: seg['start'])
        current_bubble = None
        
        for seg in segments:
            seg_text = seg.get('text', '').strip()
            if current_bubble is None:
                current_bubble = {
                    'speaker': speaker,
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg_text,
                    'overlap': ""
                }
            else:
                gap = seg['start'] - current_bubble['end']
                duration = seg['end'] - current_bubble['start']
                if gap > pause_threshold or duration > max_duration:
                    bubbles.append(current_bubble)
                    current_bubble = {
                        'speaker': speaker,
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg_text,
                        'overlap': ""
                    }
                else:
                    current_bubble['end'] = seg['end']
                    current_bubble['text'] += " " + seg_text
        
        if current_bubble is not None:
            bubbles.append(current_bubble)
    bubbles.sort(key=lambda b: b['start'])
    return bubbles

# ---------------------------------------------------------------------------
# ASYNC PROCESSING LOGIC
# ---------------------------------------------------------------------------

async def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single job by:
      - Converting each file to mono WAV
      - Sending to ASR (and optionally diarization)
      - Merging results
      - Setting job["status"] to "transcribed" or "error"
      - Updating 'transcribed_at' and 'time_taken'
    """
    temp_files = []
    job_id = job.get("job_id", "unknown")
    start_time = datetime.datetime.now()
    print(f"[process_job] Starting job_id={job_id}")

    try:
        # Extract the file paths
        orig_file_paths = ensure_list(job.get("files", []))
        print(f"[process_job] job_id={job_id}, original file paths={orig_file_paths}")

        # Calculate total size
        total_size_bytes = 0
        for path in orig_file_paths:
            if os.path.exists(path):
                total_size_bytes += os.path.getsize(path)
            else:
                print(f"[process_job] job_id={job_id}, file not found: {path}")
        job["size"] = total_size_bytes / (1024 * 1024)
        print(f"[process_job] job_id={job_id}, total size in MB={job['size']}")

        # Convert to mono WAV
        converted_paths = []
        durations = []
        for orig_path in orig_file_paths:
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_wav_path = tmp_wav.name
            tmp_wav.close()
            print(f"[process_job] Converting {orig_path} -> {tmp_wav_path}")
            try:
                dur = convert_to_mono_wav(orig_path, tmp_wav_path)
                converted_paths.append(tmp_wav_path)
                temp_files.append(tmp_wav_path)
                durations.append(dur)
            except Exception as conv_err:
                print(f"[process_job] job_id={job_id}, conversion failed: {conv_err}")

        if not converted_paths:
            raise RuntimeError("No files were successfully converted.")
        print(f"[process_job] job_id={job_id}, converted_paths={converted_paths}")
        
        # ─── sum up durations and store in job dict ───
        job["length"] = sum(durations)
        print(f"[process_job] job_id={job_id}, total audio length={job['length']}s")

        source_data = job.get("source_data", {})
        params = {}
        if source_data.get("num_speakers"):
            params["num_speakers"] = source_data["num_speakers"]

        # Determine language: use source_data["language"] if set; otherwise default to "auto"
        lang_value = source_data.get("language")
        final_language = lang_value if lang_value else "auto"

        async with httpx.AsyncClient(timeout=httpx.Timeout(ASR_TIMEOUT)) as client:
            # Build a comma-separated languages string for all files
            languages_list = [final_language] * len(converted_paths)
            languages_str = ",".join(languages_list)
            asr_task = client.post(ASR_URL, params={"languages": languages_str}, json=converted_paths)
            diar_task = None
            if source_data.get("diarization"):
                print(f"[process_job] job_id={job_id}, diarization enabled.")
                diar_task = client.post(DIAR_URL, params=params, json=converted_paths)

            if diar_task:
                asr_response, diar_response = await asyncio.gather(asr_task, diar_task)
            else:
                asr_response = await asr_task
                diar_response = None

        # Process ASR response
        print(f"[process_job] job_id={job_id}, asr_response.status={asr_response.status_code}")
        try:
            asr_data = asr_response.json()
        except Exception as e:
            print(f"[process_job] job_id={job_id}, error decoding ASR response: {e}")
            asr_data = {}

        # Process diarization response
        if diar_response:
            print(f"[process_job] job_id={job_id}, diar_response.status={diar_response.status_code}")
            try:
                diarization_data = diar_response.json()
            except Exception as e:
                print(f"[process_job] job_id={job_id}, error decoding diar response: {e}")
                diarization_data = {}
        else:
            diarization_data = {}

        # If diarization is used, call process-transcription
        if source_data.get("diarization"):
            print(f"[process_job] job_id={job_id}, merging ASR+diarization.")
            process_payload = {
                "transcription": None,
                "diarization": None
            }
            # If asr_data is { "filename.wav": [ ...segments... ] }, pick the first key
            if isinstance(asr_data, dict) and asr_data:
                first_key_asr = list(asr_data.keys())[0]
                process_payload["transcription"] = asr_data[first_key_asr]
            else:
                raise ValueError("ASR data must contain at least one key with segments.")
            
            if isinstance(diarization_data, dict) and diarization_data:
                first_key_diar = list(diarization_data.keys())[0]
                diar_data_list = diarization_data[first_key_diar].get("diarization", [])
                process_payload["diarization"] = diar_data_list
            else:
                process_payload["diarization"] = []

            async with httpx.AsyncClient(timeout=httpx.Timeout(DIAR_TIMEOUT)) as client:
                process_resp = await client.post(PROCESS_URL, json=process_payload)
            if process_resp.status_code != 200:
                print(f"[process_job] job_id={job_id}, process-transcription error: {process_resp.text}")
                raise RuntimeError("Process-transcription failed")

            try:
                process_data = process_resp.json()
            except Exception as e:
                print(f"[process_job] job_id={job_id}, error decoding process-transcription response: {e}")
                process_data = {}

            final_transcription = process_data.get("speech_bubbles", [])
        else:
            print(f"[process_job] job_id={job_id}, no diarization, beautifying ASR output.")
            # asr_data -> pick the first key
            if isinstance(asr_data, dict) and asr_data:
                first_key_asr = list(asr_data.keys())[0]
                transcription_segments = asr_data[first_key_asr]
                final_transcription = create_speech_bubbles_mono(
                    asr_transcription=transcription_segments,
                    pause_threshold=PAUSE_THRESHOLD,
                    max_duration=MAX_SPEECH_BUBBLE
                )
            else:
                final_transcription = []

        # Decide final status
        if final_transcription:
            job["status"] = "transcribed"
        else:
            job["status"] = "error"
        job["transcription"] = final_transcription
        job["transcribed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        job["time_taken"] = (datetime.datetime.now() - start_time).total_seconds()
        print(f"[process_job] job_id={job_id}, final status={job['status']}")
    except Exception as e:
        print(f"[process_job] job_id={job_id}, error: {e}")
        traceback.print_exc()
        job["status"] = "error"
    finally:
        # Cleanup temp files
        for tf in temp_files:
            try:
                os.remove(tf)
                print(f"[process_job] job_id={job_id}, removed temp file={tf}")
            except Exception as remove_err:
                print(f"[process_job] job_id={job_id}, error removing {tf}: {remove_err}")
    return job

async def process_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run `process_job` for each job concurrently.
    """
    results = await asyncio.gather(*(process_job(job) for job in jobs))
    return results

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

async def main_loop(interval: float = 10.0):
    """
    Every `interval` seconds, fetch new 'in queue' jobs, process them, and update DB.
    """
    print("[main_loop] Starting ASR worker loop.")
    while True:
        try:
            # 1) Fetch in-queue jobs
            in_queue_jobs = get_in_queue_jobs(DATABASE)
            if not in_queue_jobs:
                print("[main_loop] No new jobs in queue.")

            # 2) For example, we might only process 'mono' or 'mono-stereo' jobs
            #    or just process all in-queue. You can filter if needed:
            filtered_jobs = [j for j in in_queue_jobs if j.get('source_data', {}).get('format') in ('mono', 'mono-stereo')]
            #filtered_jobs = in_queue_jobs  # Let's just process all 'in queue'.

            if filtered_jobs:
                print(f"[main_loop] Found {len(filtered_jobs)} job(s) to process.")
                # 3) Process them
                final_results = await process_jobs(filtered_jobs)
                # 4) Update DB
                for job_dict in final_results:
                    update_job_in_db(job_dict, DATABASE)
            else:
                print("[main_loop] Nothing to process.")

        except Exception as e:
            print("[main_loop] Error:", e)
            traceback.print_exc()

        print(f"[main_loop] Sleeping for {interval} seconds.")
        await asyncio.sleep(interval)

if __name__ == "__main__":
    # Option A: just run the async main_loop forever
    # Run with: python asr_worker.py
    try:
        asyncio.run(main_loop(10.0))
    except KeyboardInterrupt:
        print("[main_loop] Stopped by user (Ctrl+C).")
