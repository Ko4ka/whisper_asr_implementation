from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import asyncio
from concurrent.futures import ProcessPoolExecutor
import warnings
from pyannote.audio import Pipeline
from pyannote.audio.utils.reproducibility import ReproducibilityWarning

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import tempfile
import os
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import traceback
import torchaudio
import multiprocessing as mp

# ---------- Константы/конфиг ----------
warnings.filterwarnings("ignore", category=ReproducibilityWarning)

MAX_FILES_PER_REQUEST = 25
CONCURRENT_PROCESSES = 1             # GPU ~1.8GB на поток — оставляем 1
TARGET_SR = 16000
GAP_STITCH_SEC = 0.75                # сшивка соседних односпикерных сегментов
MIN_SEG_DUR_SEC = 0.50               # минимальная длительность сегмента после сшивки

# Лучше хранить токен в переменных окружения
HF_TOKEN = os.getenv("HF_TOKEN", "hf_XXXX_REPLACE_ME")

# ---------- Ленивая инициализация пайплайна в каждом процессе ----------
_PIPELINE = None

def get_pipeline():
    """
    Инициализирует и кэширует пайплайн в контексте текущего процесса.
    Сделано лениво, чтобы не шарить стейт между процессами.
    """
    global _PIPELINE
    if _PIPELINE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        # Не у всех версий доступна .instantiate — закомментируй блок, если ругается,
        # либо подстрой ключи под свою версию.
        try:
            pipe = pipe.instantiate({
                "segmentation": {
                    "duration": 30.0,  # chunk duration
                    "step": 5.0,       # hop (overlap = duration - step)
                },
                # "clustering": {"method": "auto"},  # пример — по необходимости
                # "resegmentation": {"enabled": True}  # если доступно
            })
        except Exception:
            # Ок — используем дефолтные параметры
            pass

        pipe.to(device)
        _PIPELINE = pipe
    return _PIPELINE

# ---------- Вспомогательные функции ----------
def ensure_mono_16k(in_path: str) -> str:
    """
    Гарантирует mono/16k wav. Если исходник подходит — возвращает путь,
    иначе пишет временный wav и возвращает его путь (caller должен удалить).
    """
    wav, sr = torchaudio.load(in_path)
    changed = False

    # to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        changed = True

    # resample
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        wav = resampler(wav)
        changed = True

    if not changed:
        return in_path

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(tmp_path, wav, TARGET_SR)
    return tmp_path

def stitch_segments(segments: List[Dict[str, Any]],
                    gap_thr: float = GAP_STITCH_SEC,
                    min_dur: float = MIN_SEG_DUR_SEC) -> List[Dict[str, Any]]:
    """
    Сшивает соседние сегменты одного спикера: если gap <= gap_thr — объединяем.
    Удаляет очень короткие сегменты (< min_dur), по возможности присоединяя их к соседям.
    """
    if not segments:
        return segments

    # sort by start, ensure monotonicity
    segs = sorted(segments, key=lambda s: (s["start"], s["end"]))

    stitched = []
    cur = segs[0].copy()

    for s in segs[1:]:
        # enforce monotonicity
        if s["start"] < cur["end"]:
            s["start"] = cur["end"]  # жёстко обрезаем пересечение

        if s["speaker"] == cur["speaker"] and (s["start"] - cur["end"]) <= gap_thr:
            # extend current
            cur["end"] = max(cur["end"], s["end"])
        else:
            stitched.append(cur)
            cur = s.copy()
    stitched.append(cur)

    # удаляем коротышей и пытаемся прилепить их к соседям того же спикера
    cleaned: List[Dict[str, Any]] = []
    for seg in stitched:
        dur = max(0.0, seg["end"] - seg["start"])
        if dur >= min_dur or not cleaned:
            cleaned.append(seg)
        else:
            # коротыш — попробуем слить с предыдущим, если спикер совпадает
            prev = cleaned[-1]
            if prev["speaker"] == seg["speaker"] and (seg["start"] - prev["end"]) <= gap_thr:
                prev["end"] = max(prev["end"], seg["end"])
            else:
                cleaned.append(seg)

    return cleaned

def run_diarization(file_path: str,
                    file_name: str,
                    num_speakers: Optional[int]) -> Tuple[str, Dict[str, Any]]:
    """
    Ворк-функция для процесса: готовит аудио, прогоняет пайплайн и постобрабатывает сегменты.
    """
    print(f'{datetime.now()} Diarizing {file_name} Num speakers {num_speakers}')
    temp_created = None
    try:
        # гарантия моно/16к
        safe_path = ensure_mono_16k(file_path)
        if safe_path != file_path:
            temp_created = safe_path

        pipeline = get_pipeline()

        if num_speakers:
            diarization = pipeline(safe_path, num_speakers=int(num_speakers))
        else:
            diarization = pipeline(safe_path)

        # извлекаем сегменты
        raw: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw.append({
                "start": float(getattr(turn, "start", 0.0) or 0.0),
                "end": float(getattr(turn, "end", 0.0) or 0.0),
                "speaker": str(speaker)
            })

        # постобработка: сортировка/монотонность/сшивка
        stitched = stitch_segments(raw, GAP_STITCH_SEC, MIN_SEG_DUR_SEC)

        return file_name, {"diarization": stitched}
    except Exception as e:
        return file_name, {"error": f"{type(e).__name__}: {e}"}
    finally:
        if temp_created and os.path.exists(temp_created):
            try:
                os.remove(temp_created)
            except OSError:
                pass

# ---------- FastAPI ----------
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed.")

    tmp_files, filenames = [], []
    try:
        for file in files:
            tmp_file_path = await save_upload_file_tmp(file)
            tmp_files.append(tmp_file_path)
            filenames.append(file.filename)

        loop = asyncio.get_running_loop()
        tasks = []
        for idx, (tmp_file, filename) in enumerate(zip(tmp_files, filenames)):
            current_num_speakers = None
            if num_speakers is not None and len(num_speakers) > idx:
                current_num_speakers = num_speakers[idx]
            tasks.append(loop.run_in_executor(
                executor, run_diarization, tmp_file, filename, current_num_speakers
            ))
        results = await asyncio.gather(*tasks)
        output = {fname: res for fname, res in results}
        return output
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in tmp_files:
            try: os.remove(p)
            except OSError: pass

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

    loop = asyncio.get_running_loop()
    tasks = []
    for path in paths:
        filename = os.path.basename(path)
        tasks.append(loop.run_in_executor(executor, run_diarization, path, filename, num_speakers))
    results = await asyncio.gather(*tasks)
    output = {fname: res for fname, res in results}
    return output

# ---------- Исполнитель ----------
# ВАЖНО: для CUDA корректнее использовать spawn
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

executor = ProcessPoolExecutor(max_workers=CONCURRENT_PROCESSES)
