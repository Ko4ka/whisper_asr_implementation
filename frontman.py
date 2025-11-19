import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sqlite3
import json
from typing import Optional
import datetime

class JobCreate(BaseModel):
    source: str = Field(..., description="Source of the job (required)")
    files: str = Field(..., description="File Path, one only (required)")
    type: Optional[str] = None
    diarization: Optional[bool] = None
    n_speakers: Optional[int] = None
    language: Optional[str] = None
    size: Optional[str] = None
    transcription: Optional[dict] = None
    transcribed_at: Optional[str] = None
    time_taken: Optional[str] = None

class JobResponse(BaseModel):
    job_id: int
    message: str
    queue: int

app = FastAPI()
DATABASE = "./Drafts/asr_queue.db"

@app.get("/job_status/{job_id}")
def job_status(job_id: int):
    """
    Fetch the 'status' column from the 'jobs' table
    for the given job_id.
    """
    print(f"[job_status] Received request for job_id={job_id}")
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        print(f"[job_status] Connected to database '{DATABASE}' successfully")

        # Query the job by job_id
        cursor.execute("SELECT status, eta FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        
        if row is None:
            print(f"[job_status] No job found with job_id={job_id}")
            raise HTTPException(status_code=404, detail="Job not found")
        
        status = row[0]
        if status is None:
            print(f"[job_status] Job found but 'status' is NULL for job_id={job_id}")
            raise HTTPException(status_code=404, detail="No status available for this job")
        
        # Extract eta, return empty string if None
        eta = row[1] if row[1] else ""
        
        # 2. Get the total number of rows that have "in queue" status
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE status = 'in queue'
              AND job_id <= ?
            """,
            (job_id,)
        )
        count_in_queue = cursor.fetchone()
        queue_count = count_in_queue[0] if count_in_queue else 0

        print(f"[job_status] job_id={job_id}, status='{status}', queue_count={queue_count}, eta='{eta}'")
    except sqlite3.Error as e:
        msg = f"[job_status] Database error: {str(e)}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)
    finally:
        conn.close()
        print("[job_status] Database connection closed")

    # Return the status in a JSON object
    return {"status": status, "queue": queue_count, "eta": eta}

@app.post("/add_job", response_model=JobResponse)
def create_job(job: JobCreate):
    print(f"[add_job] Received request to create a job with data: {job.dict()}")
    # 1. Automatically set 'added_at' (no milliseconds)
    added_at_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. Force 'status' to be "in queue"
    status_str = "in queue"
    
    # 3. Convert JSON fields to strings if present
    settings = {}
    settings['format'] = job.type if job.type else 'mono-stereo'
    settings['diarization'] = job.diarization if job.diarization else False
    settings['num_speakers'] = job.n_speakers if job.n_speakers else False
    settings['language'] = job.language if job.language else False
    source_data_str = json.dumps(settings)
    transcription_str = json.dumps(job.transcription) if job.transcription else None
    files_str = json.dumps(job.files, ensure_ascii=False)  # required

    print("[add_job] Prepared fields for insertion into jobs table.")

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        print(f"[add_job] Connected to database '{DATABASE}' successfully.")

        # 4. Insert into DB
        insert_sql = """
        INSERT INTO jobs (
            added_at,
            source,
            source_data,
            files,
            size,
            status,
            transcription,
            transcribed_at,
            time_taken
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_sql, (
            added_at_str,
            job.source,
            source_data_str,
            files_str,
            job.size,
            status_str,
            transcription_str,
            job.transcribed_at,
            job.time_taken
        ))
        conn.commit()
        
        # 5. Get the newly inserted job's ID
        job_id = cursor.lastrowid
        
        # 6. Count how many jobs have 'in queue' AND job_id <= the current job_id
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE status = 'in queue'
              AND job_id <= ?
            """,
            (job_id,)
        )
        count_in_queue_row = cursor.fetchone()
        queue_count = count_in_queue_row[0] if count_in_queue_row else 0

        print(f"[add_job] Successfully inserted job_id={job_id}, queue_count={queue_count}")
    except sqlite3.Error as e:
        msg = f"[add_job] Database error: {str(e)}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)
    finally:
        conn.close()
        print("[add_job] Database connection closed.")

    return {
        "job_id": job_id,
        "message": "Job added successfully",
        "queue": queue_count
    }

@app.get("/get_transcription/{job_id}")
def get_transcription(job_id: int):
    """
    Fetch the 'transcription' column from the 'jobs' table
    for the given job_id.
    If found, parse and return it as JSON.
    """
    print(f"[get_transcription] Received request for job_id={job_id}")
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        print(f"[get_transcription] Connected to database '{DATABASE}' successfully.")

        # 1) Query the job by job_id for the 'transcription' column
        cursor.execute("SELECT transcription, length FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        
        if row is None:
            print(f"[get_transcription] No job found for job_id={job_id}")
            raise HTTPException(status_code=404, detail="Job not found")
        
        transcription_str, length_val = row  # JSON text + integer seconds

        if not transcription_str:
            print(f"[get_transcription] No transcription found for job_id={job_id}")
            raise HTTPException(
                status_code=404, 
                detail="No transcription found (job may still be processing)"
            )

        # 2) Parse the JSON
        try:
            transcription_data = json.loads(transcription_str)
        except json.JSONDecodeError:
            msg = "[get_transcription] Corrupted transcription data"
            print(msg)
            raise HTTPException(status_code=500, detail=msg)

        print(f"[get_transcription] Successfully fetched transcription for job_id={job_id}")
    except sqlite3.Error as e:
        msg = f"[get_transcription] Database error: {str(e)}"
        print(msg)
        raise HTTPException(status_code=500, detail=msg)
    finally:
        conn.close()
        print("[get_transcription] Database connection closed.")

    # 3) Return the transcription as JSON
    return {
        "transcription": transcription_data,
        "length": length_val
    }
