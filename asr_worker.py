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
NUM_ASR_THREADS = 2  # for correct ETA calculation

# ETA calculation constants (regression: Ŷ = 3.5473 + 0.06228X)
# Where Y = processing time in seconds, X = audio length in seconds
ETA_INTERCEPT = 3.5473
ETA_SLOPE = 0.06228

# ---------------------------------------------------------------------------
# ETA CALCULATION FUNCTIONS
# ---------------------------------------------------------------------------

def estimate_processing_seconds(audio_length_s: float) -> float:
    """
    Return predicted processing time (seconds) from regression.
    Formula: Ŷ = 3.5473 + 0.06228X
    Where Y = processing time in seconds, X = audio length in seconds
    """
    return max(0.0, ETA_INTERCEPT + ETA_SLOPE * audio_length_s)

def calculate_queue_aware_etas(
    jobs: List[Dict[str, Any]], 
    transcribing_jobs: List[Dict[str, Any]],
    num_threads: int = NUM_ASR_THREADS
) -> List[Dict[str, Any]]:
    """
    Calculate queue-aware ETAs for jobs by simulating thread pool assignment.
    Takes into account currently transcribing jobs and available slots.
    
    Args:
        jobs: List of jobs to calculate ETAs for (typically 'in queue' jobs)
        transcribing_jobs: List of jobs currently being transcribed
        num_threads: Number of concurrent processing slots
    
    Returns:
        List of jobs with updated 'eta' (datetime string) and 'eta_seconds' fields
    """
    now = datetime.datetime.now()
    
    # Initialize thread slots with current completion times
    # Each slot represents when it will be free
    thread_slots = [now] * num_threads
    
    # Process currently transcribing jobs to update slot availability
    # Sort transcribing jobs by ETA to assign them to slots properly
    trans_jobs_with_eta = []
    for trans_job in transcribing_jobs:
        eta_str = trans_job.get("eta")
        if eta_str:
            try:
                eta_dt = datetime.datetime.strptime(eta_str, "%Y-%m-%d %H:%M:%S")
                trans_jobs_with_eta.append(eta_dt)
            except (ValueError, TypeError) as e:
                print(f"[calculate_queue_aware_etas] Error parsing ETA for transcribing job: {e}")
    
    # Sort by ETA and assign each transcribing job to a unique slot
    # Each transcribing job occupies one slot until its completion time
    trans_jobs_with_eta.sort()
    for i, eta_dt in enumerate(trans_jobs_with_eta[:num_threads]):  # Only assign up to num_threads jobs
        # Assign to slot i (each transcribing job gets its own slot)
        if eta_dt > thread_slots[i]:
            thread_slots[i] = eta_dt
    
    # Sort jobs by added_at to process in order
    sorted_jobs = sorted(jobs, key=lambda j: j.get("added_at", ""))
    
    # Assign each job to the earliest available slot
    for job in sorted_jobs:
        audio_length = job.get("length", 0)
        if audio_length <= 0:
            # If length not yet calculated, skip ETA calculation
            job["eta"] = None
            job["eta_seconds"] = None
            continue
        
        # Calculate base processing time
        base_eta_seconds = estimate_processing_seconds(audio_length)
        
        # Find the earliest available slot
        earliest_slot_idx = min(range(len(thread_slots)), key=lambda i: thread_slots[i])
        slot_start_time = thread_slots[earliest_slot_idx]
        
        # Calculate when this job will complete
        completion_time = slot_start_time + datetime.timedelta(seconds=base_eta_seconds)
        
        # Update the slot to reflect this job's completion
        thread_slots[earliest_slot_idx] = completion_time
        
        # Store ETA in job dict
        job["eta"] = completion_time.strftime("%Y-%m-%d %H:%M:%S")
        job["eta_seconds"] = base_eta_seconds
    
    return sorted_jobs

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
        "length",
        "eta"
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
        # Handle case where row might have different number of columns (e.g., if eta column was added later)
        row_dict = {}
        for i, col in enumerate(columns):
            if i < len(row):
                row_dict[col] = row[i]
            else:
                row_dict[col] = None
        
        # Parse 'source_data' JSON
        if row_dict.get("source_data"):
            try:
                row_dict["source_data"] = json.loads(row_dict["source_data"])
            except json.JSONDecodeError:
                row_dict["source_data"] = None

        # Parse 'files' JSON (often a list, but user might store string)
        if row_dict.get("files"):
            try:
                row_dict["files"] = json.loads(row_dict["files"])
            except json.JSONDecodeError:
                row_dict["files"] = []

        results.append(row_dict)
    print(f"[get_in_queue_jobs] Found {len(results)} job(s) in queue.")
    return results

def get_transcribing_jobs(db_path=DATABASE) -> List[Dict[str, Any]]:
    """
    Fetch all rows with status='transcribing' from the jobs table and parse JSON fields.
    Returns a list of job dicts.
    """
    print("[get_transcribing_jobs] Checking DB for 'transcribing' jobs.")
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
        "length",
        "eta"
    ]
    try:
        cursor.execute("SELECT * FROM jobs WHERE status = 'transcribing'")
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"[get_transcribing_jobs] DB error: {e}")
        rows = []
    finally:
        conn.close()

    results = []
    for row in rows:
        # Handle case where row might have different number of columns
        row_dict = {}
        for i, col in enumerate(columns):
            if i < len(row):
                row_dict[col] = row[i]
            else:
                row_dict[col] = None
        
        # Parse 'source_data' JSON
        if row_dict.get("source_data"):
            try:
                row_dict["source_data"] = json.loads(row_dict["source_data"])
            except json.JSONDecodeError:
                row_dict["source_data"] = None

        # Parse 'files' JSON
        if row_dict.get("files"):
            try:
                row_dict["files"] = json.loads(row_dict["files"])
            except json.JSONDecodeError:
                row_dict["files"] = []

        results.append(row_dict)
    print(f"[get_transcribing_jobs] Found {len(results)} job(s) transcribing.")
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
        length = ?,
        eta = ?
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
            job.get("eta"),  # ETA datetime string
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

        # Calculate ETA before starting transcription
        # For a single job starting now, ETA is just the base processing time
        base_eta_seconds = estimate_processing_seconds(job["length"])
        start_time_dt = datetime.datetime.now()
        completion_time = start_time_dt + datetime.timedelta(seconds=base_eta_seconds)
        job["eta"] = completion_time.strftime("%Y-%m-%d %H:%M:%S")
        job["status"] = "transcribing"
        
        # Update DB with transcribing status and ETA before making ASR calls
        update_job_in_db(job, DATABASE)
        print(f"[process_job] job_id={job_id}, set status='transcribing', eta={job['eta']}")

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
        if asr_response.status_code != 200:
            print(f"[process_job] job_id={job_id}, ASR response text: {asr_response.text}")
            print(f"[process_job] job_id={job_id}, ASR response headers: {asr_response.headers}")
        try:
            asr_data = asr_response.json()
            print(f"[process_job] job_id={job_id}, asr_data keys: {list(asr_data.keys()) if isinstance(asr_data, dict) else 'not a dict'}")
            print(f"[process_job] job_id={job_id}, asr_data type: {type(asr_data)}, asr_data: {asr_data}")
        except Exception as e:
            print(f"[process_job] job_id={job_id}, error decoding ASR response: {e}")
            print(f"[process_job] job_id={job_id}, ASR response text (raw): {asr_response.text}")
            asr_data = {}

        # Process diarization response
        if diar_response:
            print(f"[process_job] job_id={job_id}, diar_response.status={diar_response.status_code}")
            if diar_response.status_code != 200:
                print(f"[process_job] job_id={job_id}, diarization response text: {diar_response.text}")
                print(f"[process_job] job_id={job_id}, diarization response headers: {diar_response.headers}")
            try:
                diarization_data = diar_response.json()
                print(f"[process_job] job_id={job_id}, diarization_data keys: {list(diarization_data.keys()) if isinstance(diarization_data, dict) else 'not a dict'}")
                print(f"[process_job] job_id={job_id}, diarization_data type: {type(diarization_data)}, diarization_data: {diarization_data}")
            except Exception as e:
                print(f"[process_job] job_id={job_id}, error decoding diar response: {e}")
                print(f"[process_job] job_id={job_id}, diarization response text (raw): {diar_response.text}")
                diarization_data = {}
        else:
            diarization_data = {}

        # If diarization is used, call process-transcription
        if source_data.get("diarization"):
            print(f"[process_job] job_id={job_id}, merging ASR+diarization.")
            print(f"[process_job] job_id={job_id}, asr_data before merge check: {asr_data}")
            print(f"[process_job] job_id={job_id}, asr_data type: {type(asr_data)}, is dict: {isinstance(asr_data, dict)}, is empty: {not asr_data if isinstance(asr_data, dict) else 'N/A'}")
            process_payload = {
                "transcription": None,
                "diarization": None
            }
            # If asr_data is { "filename.wav": [ ...segments... ] }, pick the first key
            if isinstance(asr_data, dict) and asr_data:
                first_key_asr = list(asr_data.keys())[0]
                process_payload["transcription"] = asr_data[first_key_asr]
            else:
                print(f"[process_job] job_id={job_id}, ERROR: ASR data is invalid. asr_data={asr_data}, type={type(asr_data)}")
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
    Also updates ETAs for jobs in queue and currently transcribing jobs.
    """
    print("[main_loop] Starting ASR worker loop.")
    while True:
        try:
            # 1) Fetch in-queue jobs
            in_queue_jobs = get_in_queue_jobs(DATABASE)
            
            # 2) Fetch currently transcribing jobs
            transcribing_jobs = get_transcribing_jobs(DATABASE)
            
            # 3) Calculate ETAs for in-queue jobs (queue-aware)
            # First, we need to get audio lengths for jobs that don't have them yet
            # For now, we'll calculate ETAs only for jobs that already have length
            # Jobs without length will get ETA calculated in process_job after conversion
            jobs_with_length = [j for j in in_queue_jobs if j.get("length") and j.get("length") > 0]
            if jobs_with_length:
                jobs_with_etas = calculate_queue_aware_etas(jobs_with_length, transcribing_jobs, NUM_ASR_THREADS)
                # Update DB with calculated ETAs
                for job in jobs_with_etas:
                    if job.get("eta"):
                        update_job_in_db(job, DATABASE)
                        print(f"[main_loop] Updated ETA for job_id={job['job_id']}: {job['eta']}")

            # 4) Filter jobs to process (mono or mono-stereo format)
            filtered_jobs = [j for j in in_queue_jobs if j.get('source_data', {}).get('format') in ('mono', 'mono-stereo')]
            # Limit to NUM_ASR_THREADS to respect concurrency
            available_slots = NUM_ASR_THREADS - len(transcribing_jobs)
            if available_slots > 0:
                filtered_jobs = filtered_jobs[:available_slots]

            if filtered_jobs:
                print(f"[main_loop] Found {len(filtered_jobs)} job(s) to process (available slots: {available_slots}).")
                # 5) Process them
                final_results = await process_jobs(filtered_jobs)
                # 6) Update DB (process_job already updates status to transcribing, but we update again for final status)
                for job_dict in final_results:
                    update_job_in_db(job_dict, DATABASE)
            else:
                if not in_queue_jobs:
                    print("[main_loop] No new jobs in queue.")
                elif available_slots <= 0:
                    print(f"[main_loop] All {NUM_ASR_THREADS} slots are busy with transcribing jobs.")
                else:
                    print("[main_loop] No jobs matching filter criteria.")

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
