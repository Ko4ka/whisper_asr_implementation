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
from typing import List, Optional, Dict, Any
from datetime import datetime
import traceback
import torchaudio
import warnings

# -------------------- Config --------------------
warnings.filterwarnings("ignore", category=ReproducibilityWarning)

MAX_FILES_PER_REQUEST = 25           # Maximum number of files per request
CONCURRENT_PROCESSES = 1             # ~1.8 GB GPU per stream
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Post-processing knobs
GAP_STITCH_SEC       = 0.50          # merge adjacent same-speaker segments with gaps ≤ this
MIN_TURN_SEC         = 0.50          # drop/merge turns shorter than this when possible
MIN_CLUSTER_SEC      = 3.0           # absorb speakers whose total duration < this
MIN_CLUSTER_RATIO    = 0.01          # or < 1% of file duration
SAD_MIN_HANGOVER_SEC = 0.35          # request SAD min_duration_on/off (may be ignored if unsupported)

# HuggingFace token (prefer env variable)
HF_TOKEN = os.getenv("HF_TOKEN", "hf_XXXX_REPLACE_ME")

# -------------------- Pipeline (instantiate with resegmentation + SAD hints) --------------------
def build_pipeline() -> Pipeline:
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=HF_TOKEN)

    # Try to configure segmentation window/step, SAD hangover, and enable resegmentation.
    # Different pyannote builds expose different knobs; we try a few variants and ignore failures.
    tried = False
    try:
        pipe = pipe.instantiate({
            "segmentation": {
                "duration": 30.0,
                "step": 5.0,
                # Some builds accept these directly:
                "min_duration_on": SAD_MIN_HANGOVER_SEC,
                "min_duration_off": SAD_MIN_HANGOVER_SEC,
            },
            "resegmentation": {"enabled": False}
        })
        tried = True
    except Exception:
        pass

    if not tried:
        try:
            # Alternate shape: put hangover under "binarize" subkey
            pipe = pipe.instantiate({
                "segmentation": {
                    "duration": 30.0,
                    "step": 5.0,
                    "binarize": {
                        "min_duration_on": SAD_MIN_HANGOVER_SEC,
                        "min_duration_off": SAD_MIN_HANGOVER_SEC,
                    },
                },
                "resegmentation": {"enabled": True}
            })
        except Exception:
            # Fall back to defaults if none of the above are available
            pass

    pipe.to(DEVICE)
    return pipe

# Build at import; in spawned processes this module is re-imported so each child gets its own
pipeline = build_pipeline()

# -------------------- Post-processing --------------------
def stitch_segments(segments: List[Dict[str, Any]],
                    gap_thr: float = GAP_STITCH_SEC,
                    min_turn: float = MIN_TURN_SEC) -> List[Dict[str, Any]]:
    """
    Merge adjacent same-speaker segments separated by small gaps and
    absorb very short turns when possible.
    """
    if not segments:
        return segments

    segs = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    merged: List[Dict[str, Any]] = []

    cur = segs[0].copy()
    for s in segs[1:]:
        s = s.copy()
        # enforce monotonic boundaries
        if s["start"] < cur["end"]:
            s["start"] = cur["end"]

        if s["speaker"] == cur["speaker"] and (s["start"] - cur["end"]) <= gap_thr:
            cur["end"] = max(cur["end"], s["end"])
        else:
            merged.append(cur)
            cur = s
    merged.append(cur)

    # absorb sub-min turns into neighbors when possible
    out: List[Dict[str, Any]] = []
    for seg in merged:
        dur = max(0.0, float(seg["end"]) - float(seg["start"]))
        if dur >= min_turn or not out:
            out.append(seg)
        else:
            prev = out[-1]
            if prev["speaker"] == seg["speaker"] and (seg["start"] - prev["end"]) <= gap_thr:
                prev["end"] = max(prev["end"], seg["end"])
            else:
                out.append(seg)
    return out

def absorb_micro_clusters(segments: List[Dict[str, Any]],
                          file_dur: float,
                          min_total: float = MIN_CLUSTER_SEC,
                          min_ratio: float = MIN_CLUSTER_RATIO,
                          gap_thr: float = GAP_STITCH_SEC) -> List[Dict[str, Any]]:
    """
    Reassign segments from "micro" speakers (by total duration) to the nearest
    neighboring major speaker, then restitch.
    """
    if not segments:
        return segments

    # total durations per speaker
    totals: Dict[str, float] = {}
    for s in segments:
        totals.setdefault(s["speaker"], 0.0)
        totals[s["speaker"]] += max(0.0, float(s["end"]) - float(s["start"]))

    tiny = {spk for spk, tot in totals.items()
            if tot < min_total or tot < (min_ratio * file_dur)}
    if not tiny:
        return segments

    segs = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))

    # Precompute totals for tie-breaking
    major_totals = {spk: tot for spk, tot in totals.items() if spk not in tiny}

    def nearest_neighbor_speaker(idx: int) -> Optional[str]:
        left_idx = None
        for j in range(idx - 1, -1, -1):
            if segs[j]["speaker"] not in tiny:
                left_idx = j
                break
        right_idx = None
        for j in range(idx + 1, len(segs)):
            if segs[j]["speaker"] not in tiny:
                right_idx = j
                break

        # If only one side exists, take it
        if left_idx is None and right_idx is None:
            return None
        if left_idx is None:
            return segs[right_idx]["speaker"]
        if right_idx is None:
            return segs[left_idx]["speaker"]

        # Compare temporal proximity
        left_gap = max(0.0, float(segs[idx]["start"]) - float(segs[left_idx]["end"]))
        right_gap = max(0.0, float(segs[right_idx]["start"]) - float(segs[idx]["end"]))

        if left_gap < right_gap:
            return segs[left_idx]["speaker"]
        if right_gap < left_gap:
            return segs[right_idx]["speaker"]

        # Tie-breaker: larger total duration cluster
        left_spk = segs[left_idx]["speaker"]
        right_spk = segs[right_idx]["speaker"]
        return left_spk if major_totals.get(left_spk, 0.0) >= major_totals.get(right_spk, 0.0) else right_spk

    # Reassign tiny clusters
    for i, s in enumerate(segs):
        if s["speaker"] in tiny:
            repl = nearest_neighbor_speaker(i)
            if repl is not None:
                s["speaker"] = repl

    # Final restitch to merge any new adjacencies
    return stitch_segments(segs, gap_thr=gap_thr, min_turn=MIN_TURN_SEC)

# -------------------- Core diarization worker --------------------
def diarize_file_process(file_path: str, file_name: str, num_speakers: Optional[int]):
    """
    Process an audio file for speaker diarization with resegmentation,
    stitching, and micro-cluster absorption.
    """
    print(f'{datetime.now()} Diarizing {file_name} Num speakers {num_speakers}')
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        num_channels = waveform.shape[0]
        if num_channels != 1:
            return file_name, {"error": "Error: Audio is not mono. Please provide a mono audio file."}

        # Run pyannote pipeline (resegmentation enabled if available)
        if num_speakers:
            diarization = pipeline(file_path, num_speakers=int(num_speakers))
        else:
            diarization = pipeline(file_path)

        # Collect raw segments
        results: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": float(getattr(turn, "start", 0.0) or 0.0),
                "end": float(getattr(turn, "end", 0.0) or 0.0),
                "speaker": str(speaker),
            })

        if not results:
            return file_name, {"diarization": []}

        # Sort & post-process
        results = sorted(results, key=lambda r: (r["start"], r["end"]))
        file_dur = float(waveform.shape[1]) / float(sample_rate)

        results = stitch_segments(results,
                                  gap_thr=GAP_STITCH_SEC,
                                  min_turn=MIN_TURN_SEC)
        results = absorb_micro_clusters(results,
                                        file_dur=file_dur,
                                        min_total=MIN_CLUSTER_SEC,
                                        min_ratio=MIN_CLUSTER_RATIO,
                                        gap_thr=GAP_STITCH_SEC)

        return file_name, {"diarization": results}
    except Exception as e:
        return file_name, {"error": str(e)}

# -------------------- FastAPI --------------------
app = FastAPI()

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        contents = await upload_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(contents)
            return tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diarize_audio_bulk")
async def diarize_audio(
    files: List[UploadFile] = File(...),
    num_speakers: Optional[List[Optional[int]]] = Form(None)
):
    try:
        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(status_code=400, detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed.")

        tmp_files: List[str] = []
        filenames: List[str] = []
        for file in files:
            tmp_file_path = await save_upload_file_tmp(file)
            tmp_files.append(tmp_file_path)
            filenames.append(file.filename)

        loop = asyncio.get_event_loop()
        tasks = []
        for idx, (tmp_file, filename) in enumerate(zip(tmp_files, filenames)):
            current_num_speakers = None
            if num_speakers is not None and len(num_speakers) > idx:
                current_num_speakers = num_speakers[idx]
            task = loop.run_in_executor(
                executor, diarize_file_process, tmp_file, filename, current_num_speakers
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        output = {filename: result for filename, result in results}

        # Clean up
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass

        return output
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diarize_audio_bulk_local")
async def diarize_audio_bulk_local(
    paths: List[str],
    num_speakers: Optional[int] = None
):
    if len(paths) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed.")

    for path in paths:
        if not os.path.isfile(path):
            raise HTTPException(status_code=400, detail=f"File does not exist or is not accessible: {path}")

    loop = asyncio.get_event_loop()
    tasks = []
    for path in paths:
        filename = os.path.basename(path)
        tasks.append(loop.run_in_executor(
            executor, diarize_file_process, path, filename, num_speakers
        ))

    results = await asyncio.gather(*tasks)
    output = {filename: result for filename, result in results}
    return output

# -------------------- Executor --------------------
executor = ProcessPoolExecutor(max_workers=CONCURRENT_PROCESSES)
