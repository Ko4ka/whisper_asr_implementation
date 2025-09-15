from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback
import random


# Globals
MAX_SPEECH_BUBBLE = 15.0
PAUSE_THRESHOLD = 1

# Initialize FastAPI
app = FastAPI()

def align_transcription_with_diarization(transcription, diarization, overlap_threshold=0.1, mode="loose"):
    """
    Aligns transcription words with diarization segments, ensuring each word is
    aligned with all segments it significantly overlaps with in "strict" mode,
    or to the speaker with the highest overlap in "loose" mode.
    """
    # Flatten the word list from the transcription data
    words = []
    for speaker, segments in transcription.items():
        for segment in segments:
            words.extend(segment['words'])
    
    aligned_words = []
    
    # Loop over each word to align it with diarization segments
    for word in words:
        word_start = word['start']
        word_end = word['end']
        word_duration = word_end - word_start
        word_text = word['word']
    
        # Keep track of overlaps with each speaker
        overlaps = []
    
        # Compare the word against all diarization segments
        for diarization_segment in diarization:
            segment_start = diarization_segment['start']
            segment_end = diarization_segment['end']
            segment_speaker = diarization_segment['speaker']
    
            # Calculate overlap
            overlap_start = max(word_start, segment_start)
            overlap_end = min(word_end, segment_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            # Calculate overlap percentage
            overlap_percentage = overlap_duration / word_duration if word_duration > 0 else 0
    
            if overlap_percentage >= overlap_threshold:
                overlaps.append({
                    'speaker': segment_speaker,
                    'overlap_percentage': overlap_percentage
                })
        
        # Strict mode: assign word to all speakers with significant overlap
        if mode == "strict":
            for overlap in overlaps:
                aligned_word = {
                    'word': word_text,
                    'start': word_start,
                    'end': word_end,
                    'speaker': overlap['speaker']
                }
                aligned_words.append(aligned_word)
        
        # Loose mode: assign word to the speaker with the highest overlap
        elif mode == "loose":
            if overlaps:
                best_match = max(overlaps, key=lambda x: x['overlap_percentage'])
                aligned_word = {
                    'word': word_text,
                    'start': word_start,
                    'end': word_end,
                    'speaker': best_match['speaker']
                }
                aligned_words.append(aligned_word)
    
    return aligned_words


def create_speech_bubbles_t(transcription, pause_threshold=PAUSE_THRESHOLD, max_duration=MAX_SPEECH_BUBBLE):
    speech_bubbles = []
    speaker_bubbles = {}  # Holds the current bubble for each speaker
    last_end_times = {}   # Tracks the last end time for each speaker

    # Ensure transcription is sorted by start time
    transcription.sort(key=lambda x: x['start'])

    for word_data in transcription:
        word = word_data['word']
        start_time = word_data['start']
        end_time = word_data['end']
        speaker = word_data['speaker']

        # Initialize the current bubble for the speaker if not already present
        if speaker not in speaker_bubbles:
            speaker_bubbles[speaker] = {"speaker": speaker, "start": None, "end": None, "text": "", "overlap": ""}
            last_end_times[speaker] = None

        current_bubble = speaker_bubbles[speaker]
        last_end_time = last_end_times[speaker]

        # If the current bubble is empty, initialize it with the current word
        if current_bubble["start"] is None:
            current_bubble["start"] = start_time
            current_bubble["end"] = end_time
            current_bubble["text"] = word
        else:
            # Check if we need to start a new bubble
            has_long_pause = last_end_time and (start_time - last_end_time > pause_threshold)
            exceeds_max_duration = (end_time - current_bubble["start"]) > max_duration

            if has_long_pause or exceeds_max_duration:
                # Finalize the current bubble and start a new one
                speech_bubbles.append(current_bubble)
                speaker_bubbles[speaker] = {
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": word,
                    "overlap": ""
                }
                current_bubble = speaker_bubbles[speaker]
            else:
                # Continue the current bubble
                current_bubble["text"] += "" + word
                current_bubble["end"] = end_time

        # Update the last end time for the speaker
        last_end_times[speaker] = end_time

    # Append any remaining bubbles
    for bubble in speaker_bubbles.values():
        if bubble["start"] is not None:
            speech_bubbles.append(bubble)

    # Sort the bubbles by start time
    speech_bubbles.sort(key=lambda x: x['start'])

    # Detect overlaps between bubbles of different speakers and capture exact sequences
    for i in range(len(speech_bubbles)):
        bubble_i = speech_bubbles[i]
        for j in range(i + 1, len(speech_bubbles)):
            bubble_j = speech_bubbles[j]
            # Stop checking if the next bubble starts after the current bubble ends
            if bubble_j['start'] > bubble_i['end']:
                break
            # Check if bubbles are from different speakers and overlap
            if bubble_i['speaker'] != bubble_j['speaker']:
                # Check for overlap
                start_i, end_i = bubble_i['start'], bubble_i['end']
                start_j, end_j = bubble_j['start'], bubble_j['end']
                if start_i < end_j and start_j < end_i:
                    # Identify exact overlapping sequences
                    words_i = bubble_i['text'].split()
                    words_j = bubble_j['text'].split()
                    overlap_sequence = []

                    # Compare sequences of words
                    for idx_i, word_i in enumerate(words_i):
                        for idx_j, word_j in enumerate(words_j):
                            if word_i == word_j:
                                temp_sequence = []
                                k = 0
                                # Check for a sequence match
                                while (
                                    idx_i + k < len(words_i)
                                    and idx_j + k < len(words_j)
                                    and words_i[idx_i + k] == words_j[idx_j + k]
                                ):
                                    temp_sequence.append(words_i[idx_i + k])
                                    k += 1
                                if len(temp_sequence) > len(overlap_sequence):
                                    overlap_sequence = temp_sequence

                    if overlap_sequence:
                        overlap_text = " ".join(overlap_sequence)
                        bubble_i['overlap'] = overlap_text
                        bubble_j['overlap'] = overlap_text

    return speech_bubbles

def generate_html_with_media_player_t(speech_bubbles, audio_file_url):

    # Calculate total speaking duration for each speaker
    speaker_durations = {}
    for bubble in speech_bubbles:
        speaker = bubble["speaker"]
        duration = bubble["end"] - bubble["start"]
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

    # Sort speakers by speaking duration
    sorted_speakers = sorted(speaker_durations.keys(), key=lambda s: -speaker_durations[s])

    # Generate distinct color palettes for speakers
    random.seed(42)
    speaker_colors = {
        speaker: {
            "background": f"hsl({random.randint(0, 360)}, {random.randint(60, 80)}%, {random.randint(80, 90)}%)",
            "text": f"hsl({random.randint(0, 360)}, {random.randint(60, 80)}%, {random.randint(20, 30)}%)"
        }
        for speaker in sorted_speakers
    }

    # Map speakers to positions (right or left)
    speaker_positions = {speaker: "right" if i % 2 == 0 else "left" for i, speaker in enumerate(sorted_speakers)}

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Speech Bubbles with Media Player</title>
        <link rel="stylesheet" href="https://cdn.plyr.io/3.7.8/plyr.css" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                color: #333;
                padding: 20px;
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .sticky-player {{
                position: fixed;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 1000;
                width: 90%;
                max-width: 600px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 10px;
            }}
            .content {{
                margin-top: 160px; /* Space for the sticky player */
                width: 90%;
                max-width: 800px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .button-row {{
                display: flex;
                gap: 10px;
                align-items: center;
                justify-content: space-between;
            }}
            #save-button {{
                padding: 5px 10px;
                font-size: 0.9em;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            #search-bar {{
                flex-grow: 1;
                display: flex;
            }}
            #search-bar input {{
                width: 100%;
                padding: 5px;
                font-size: 0.9em;
                border: 1px solid #ccc;
                border-radius: 5px;
            }}
            #name-form {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
            }}
            #name-form input {{
                padding: 5px;
                font-size: 0.9em;
                border: 1px solid #ccc;
                border-radius: 5px;
            }}
            #name-form button {{
                padding: 5px 10px;
                font-size: 0.9em;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            .bubble-container {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .bubble {{
                border-radius: 10px;
                padding: 15px;
                max-width: 70%;
                word-wrap: break-word;
                cursor: pointer;
            }}
            .bubble.right {{
                align-self: flex-end;
            }}
            .bubble.left {{
                align-self: flex-start;
            }}
            .timestamp {{
                font-size: 0.85em;
                color: #555;
                margin-top: 5px;
                text-align: right;
            }}
            .speaker-name {{
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="sticky-player">
            <audio id="player" controls>
                <source src="{audio_file_url}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>

        <div class="content">
            <!-- Speaker Name Assignment Form -->
            <form id="name-form">
                {''.join(f'<input type="text" id="{speaker}" placeholder="Имя: {speaker}" />' for speaker in sorted_speakers)}
                <button type="button" onclick="updateNames()">Обновить имена</button>
            </form>

            <!-- Button Row -->
            <div class="button-row">
                <button id="save-button">Скачать текст</button>
                <div id="search-bar">
                    <input type="text" id="search-input" placeholder="Поиск по тексту..." oninput="filterBubbles()" />
                </div>
            </div>

            <div class="bubble-container">
    """

    # Add bubbles for each speech segment
    for bubble in speech_bubbles:
        speaker = bubble["speaker"]
        colors = speaker_colors[speaker]
        position = speaker_positions[speaker]
        start_time = format_time(bubble["start"])
        end_time = format_time(bubble["end"])

        html_content += f"""
        <div class="bubble {position}" style="background-color: {colors['background']}; color: {colors['text']};" data-start="{bubble["start"]}">
            <div class="text"><span class="speaker-name" id="name-{speaker}">{speaker}</span>: {bubble["text"]}</div>
            <div class="timestamp">[{start_time} - {end_time}]</div>
        </div>
        """

    # Close the HTML structure
    html_content += """
            </div>
        </div>

        <script src="https://cdn.plyr.io/3.7.8/plyr.polyfilled.js"></script>
        <script>
            const player = new Plyr('#player', {
                controls: ['play', 'progress', 'current-time', 'duration', 'mute', 'volume']
            });

            // Add event listeners to bubbles for seeking
            document.querySelectorAll('.bubble').forEach(bubble => {
                bubble.addEventListener('click', () => {
                    const startTime = parseFloat(bubble.dataset.start);
                    player.currentTime = startTime;
                    player.play();
                });
            });

            // Update speaker names dynamically
            function updateNames() {
                document.querySelectorAll('#name-form input').forEach(input => {
                    const speakerId = input.id;
                    const name = input.value.trim();
                    if (name) {
                        document.querySelectorAll(`#name-${speakerId}`).forEach(el => {
                            el.textContent = name;
                        });
                    }
                });
            }

            // Save as TXT functionality
            document.getElementById('save-button').addEventListener('click', () => {
                const bubbles = document.querySelectorAll('.bubble');
                let textContent = '';

                bubbles.forEach(bubble => {
                    const speaker = bubble.querySelector('.speaker-name').textContent;
                    const text = bubble.querySelector('.text').textContent;
                    const timestamp = bubble.querySelector('.timestamp').textContent;
                    textContent += `${speaker} (${timestamp}): ${text}\n\n`;
                });

                const blob = new Blob([textContent], { type: 'text/plain' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = 'transcription.txt';
                link.click();
            });

            // Search functionality
            function filterBubbles() {
                const query = document.getElementById('search-input').value.toLowerCase();
                const bubbles = document.querySelectorAll('.bubble');
                bubbles.forEach(bubble => {
                    const text = bubble.querySelector('.text').textContent.toLowerCase();
                    bubble.style.display = text.includes(query) ? '' : 'none';
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content


def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{mins}:{secs:02}.{ms:03}"




@app.post("/generate-html")
async def generate_html(
    bubbles: List[Dict[str, Any]],
    audio_file_url: str = Query(..., description="URL of the audio file")
):
    """
    Receives the bubbles object and a file path as a string,
    runs `generate_html_with_media_player_t`.
    """
    try:
        html_content = generate_html_with_media_player_t(bubbles, audio_file_url)
        return {"html": html_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- helpers for readable text ----------------------------------------------

# --- spacing helpers ---
OPENERS = {'(', '[', '«', '“', '"', '‘', "'"}
CLOSERS = {')', ']', '»', '”', '"', '’', "'"}
NO_SPACE_BEFORE = {',', '.', '!', '?', ':', ';', '…', '%', ')', ']', '»', '”', '"', '’', "'"}
NO_SPACE_AFTER  = {'(', '[', '«', '“', '"', '‘', "'"}
NO_SPACE_AROUND = {"'", '’', '-', '–', '—'}

def _append_token(text: str, token: str) -> str:
    if not text:
        return token
    prev = text[-1]
    sep = ' '
    if token in NO_SPACE_BEFORE or token in CLOSERS: sep = ''
    if prev in NO_SPACE_AFTER or prev in OPENERS:    sep = ''
    if prev in NO_SPACE_AROUND or token in NO_SPACE_AROUND: sep = ''
    return (text + sep + token).replace(' \n', '\n')

def _clean_text(s: str) -> str:
    import re
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s+([,.!?;:%\)\]\»])', r'\1', s)
    s = re.sub(r'([\(\[\«])\s+', r'\1', s)
    return s.strip()

# --- new bubble builder ------------------------------------------------------

from typing import List, Dict, Any
from math import isfinite

def flatten_words_absolute(transcription: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Returns a chronological list of {'word','start','end'}.
    If words already absolute (e.g., > segment len), keeps them as is.
    If they look relative to the segment, adds seg.start.
    """
    out = []
    for _spk, segs in transcription.items():
        for seg in segs:
            s0 = float(seg.get('start', 0.0) or 0.0)
            s1 = float(seg.get('end', s0) or s0)
            dur = max(0.0, s1 - s0)
            for w in seg.get('words', []):
                ws = float(w.get('start', 0.0) or 0.0)
                we = float(w.get('end', ws) or ws)
                token = str(w.get('word', '') or '').strip()
                if not token: continue
                # heuristic: if within seg duration, treat as relative
                if dur > 0 and ws <= dur + 1e-6 and we <= dur + 1e-6:
                    ws_abs, we_abs = s0 + ws, s0 + we
                else:
                    ws_abs, we_abs = ws, we
                if not (isfinite(ws_abs) and isfinite(we_abs)) or we_abs <= ws_abs:
                    continue
                out.append({'word': token, 'start': ws_abs, 'end': we_abs})
    out.sort(key=lambda x: (x['start'], x['end']))
    return out

def merge_diar_turns(diar: List[Dict[str, Any]], gap_thr: float = 0.3) -> List[Dict[str, Any]]:
    """
    Merge adjacent same-speaker turns if the gap <= gap_thr.
    Keeps overlaps between different speakers (they'll render as concurrent bubbles).
    """
    if not diar: return []
    segs = sorted(diar, key=lambda s: (s['start'], s['end']))
    out = []
    cur = dict(segs[0])
    for s in segs[1:]:
        if s['speaker'] == cur['speaker'] and s['start'] - cur['end'] <= gap_thr:
            cur['end'] = max(cur['end'], s['end'])
        else:
            out.append(cur)
            cur = dict(s)
    out.append(cur)
    # enforce monotonic
    for i in range(1, len(out)):
        if out[i]['start'] < out[i-1]['end']:
            out[i]['start'] = out[i-1]['end']
    return out


def create_speech_bubbles_from_turns(
    words_abs: List[Dict[str, Any]],
    diarization: List[Dict[str, Any]],
    pause_threshold: float = PAUSE_THRESHOLD,
    max_duration: float = MAX_SPEECH_BUBBLE
) -> List[Dict[str, Any]]:
    """
    Build bubbles by diarization turns (speaker segments).
    For each diar turn, collect words whose midpoint lies inside the turn.
    If a turn is long, split by pauses or by max_duration.
    """
    if not words_abs or not diarization:
        return []

    turns = merge_diar_turns(diarization, gap_thr=0.3)
    bubbles: List[Dict[str, Any]] = []

    # index words by time for linear scan
    w = 0
    n = len(words_abs)

    for t in turns:
        ts, te, spk = float(t['start']), float(t['end']), str(t['speaker'])

        # collect words whose midpoint is inside [ts, te]
        turn_words: List[Dict[str, Any]] = []
        # advance to first possibly overlapping word
        while w < n and words_abs[w]['end'] <= ts:
            w += 1
        j = w
        while j < n and words_abs[j]['start'] < te:
            mid = 0.5 * (words_abs[j]['start'] + words_abs[j]['end'])
            if ts <= mid <= te:
                turn_words.append(words_abs[j])
            j += 1

        if not turn_words:
            continue

        # split into readable bubbles inside the turn
        cur_text = ""
        cur_start = turn_words[0]['start']
        cur_end   = turn_words[0]['end']

        def flush():
            nonlocal cur_text, cur_start, cur_end
            if cur_text:
                bubbles.append({
                    "speaker": spk,
                    "start": cur_start,
                    "end":   cur_end,
                    "text":  _clean_text(cur_text),
                    "overlap": ""
                })
                cur_text = ""

        for k in range(len(turn_words)):
            token = turn_words[k]['word']
            ws, we = turn_words[k]['start'], turn_words[k]['end']

            # check split conditions
            long_pause = (k > 0) and (ws - cur_end > pause_threshold)
            too_long   = (we - cur_start) > max_duration

            if long_pause or too_long:
                flush()
                cur_start, cur_end = ws, we
                cur_text = token
            else:
                if not cur_text:
                    cur_text = token
                    cur_start, cur_end = ws, we
                else:
                    cur_text = _append_token(cur_text, token)
                    cur_end  = we

        flush()

    # chronological sort (turn order already chronological, but safe)
    bubbles.sort(key=lambda b: (b['start'], b['end']))

    # enforce monotone starts (avoid tiny reversals after merges)
    for i in range(1, len(bubbles)):
        if bubbles[i]['start'] < bubbles[i-1]['end']:
            bubbles[i]['start'] = bubbles[i-1]['end']

    return bubbles

    
@app.post("/process-transcription")
async def process_transcription(
    transcription: Dict[str, List[Dict[str, Any]]],
    diarization: List[Dict[str, Any]]
):
    try:
        words_abs = flatten_words_absolute(transcription)  # keeps absolute if already absolute
        speech_bubbles = create_speech_bubbles_from_turns(
            words_abs,
            diarization,
            pause_threshold=PAUSE_THRESHOLD,
            max_duration=MAX_SPEECH_BUBBLE
        )
        return {"speech_bubbles": speech_bubbles}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

