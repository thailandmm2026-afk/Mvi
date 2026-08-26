import os
import time
import uuid
import asyncio
import subprocess
import re
import shutil
from pathlib import Path
from datetime import timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import yt_dlp
import whisper
from deep_translator import GoogleTranslator
import edge_tts

# ========== CONFIG ==========
TEMP = Path("temp_files")
TEMP.mkdir(exist_ok=True)
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

VOICES = {
    "thiha": "my-MM-ThihaNeural",
    "nilar": "my-MM-NilarNeural",
}

whisper_model = None
jobs = {}  # job_id -> status dict

app = FastAPI(title="Myanmar Voice Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    p = Path("index.html")
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>")

# ========== HELPERS ==========
def set_status(job_id: str, step: str, detail: str = "", percent: int = 0, done: bool = False, error: str = None, result: dict = None):
    jobs[job_id] = {
        "step": step,
        "detail": detail,
        "percent": percent,
        "done": done,
        "error": error,
        "result": result,
    }

def get_duration(path: str):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return None

def load_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper tiny...")
        whisper_model = whisper.load_model("tiny")
        print("Whisper ready.")
    return whisper_model

def translate_text(text: str) -> str:
    try:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        translator = GoogleTranslator(source="en", target="my")
        out = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            try:
                t = translator.translate(s)
                out.append(t or s)
            except Exception:
                out.append(s)
            time.sleep(0.04)
        result = " ".join(out)
        result = result.replace("ငါ", "ကျွန်တော်").replace("မင်း", "ခင်ဗျား").replace("သင်", "ခင်ဗျား")
        return re.sub(r"\s+", " ", result).strip()
    except Exception:
        return text

def split_sentences(text: str):
    parts = re.split(r"(?<=[။။\?!\.])", text)
    parts = [s.strip() for s in parts if s.strip() and len(s) >= 2]
    return parts if parts else [text.strip()]

def format_ts(td: timedelta) -> str:
    total = td.total_seconds()
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    ms = int((total % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(text: str, duration: float) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    weights = []
    for s in sentences:
        my = sum(1 for c in s if "\u1000" <= c <= "\u109F")
        weights.append(my * 1.2 + (len(s) - my))
    total_w = sum(weights) or 1
    speech = duration * 0.9
    start = timedelta(0)
    lines = []
    for i, sent in enumerate(sentences):
        dur = max(speech * (weights[i] / total_w), 0.7)
        end = start + timedelta(seconds=dur)
        lines += [str(i + 1), f"{format_ts(start)} --> {format_ts(end)}", sent, ""]
        start = end
    return "\n".join(lines)

def speed_to_rate(speed: float) -> str:
    p = round((speed - 1.0) * 100)
    return f"+{p}%" if p >= 0 else f"{p}%"

async def tts(text: str, outfile: str, voice_key: str, speed: float):
    voice_id = VOICES.get(voice_key, VOICES["thiha"])
    rate = speed_to_rate(speed)
    communicate = edge_tts.Communicate(text, voice_id, rate=rate)
    await communicate.save(outfile)

def download_video(url: str, out_path: str) -> Optional[str]:
    opts = {
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 300,
        "retries": 3,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            return None
        fname = ydl.prepare_filename(info)
        if os.path.exists(fname) and os.path.getsize(fname) > 50000:
            return fname
    folder = os.path.dirname(out_path)
    for f in os.listdir(folder):
        if f.endswith(".mp4") and os.path.getsize(os.path.join(folder, f)) > 50000:
            return os.path.join(folder, f)
    return None

def transcribe(video_path: str) -> Optional[str]:
    model = load_whisper()
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    cmd = [
        "ffmpeg", "-i", video_path,
        "-acodec", "libmp3lame", "-ab", "128k",
        "-ar", "16000", "-ac", "1", audio_path, "-y"
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    if r.returncode != 0 or not os.path.exists(audio_path):
        return None
    result = model.transcribe(audio_path, language="my", fp16=False)
    try:
        os.remove(audio_path)
    except Exception:
        pass
    return (result.get("text") or "").strip()

def combine(video_path: str, audio_path: str, final_path: str) -> bool:
    v_dur = get_duration(video_path) or 30
    a_dur = get_duration(audio_path) or 30
    speed = v_dur / a_dur if a_dur > 0 else 1.0
    temp_vid = str(TEMP / f"tmp_{uuid.uuid4().hex[:8]}.mp4")
    if abs(speed - 1.0) > 0.02:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-filter:v", f"setpts={1/speed}*PTS",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-y", temp_vid
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(temp_vid):
            temp_vid = video_path
    else:
        temp_vid = video_path

    cmd = [
        "ffmpeg", "-i", temp_vid, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", final_path
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        cmd = [
            "ffmpeg", "-i", temp_vid, "-i", audio_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", final_path
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300)

    if temp_vid != video_path and os.path.exists(temp_vid):
        try:
            os.remove(temp_vid)
        except Exception:
            pass
    return r.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 1000

# ========== BACKGROUND JOB ==========
async def run_job(job_id: str, url: str, voice: str, speed: float, mode: str, file_path: str = None):
    work = TEMP / job_id
    work.mkdir(exist_ok=True)
    video_path = None

    try:
        # 1. Video
        set_status(job_id, "video", "🎬 Video ရယူနေပါသည်...", 10)
        if file_path and os.path.exists(file_path):
            video_path = file_path
        elif url:
            out_tmpl = str(work / "dl.%(ext)s")
            video_path = download_video(url, out_tmpl)
            if not video_path:
                set_status(job_id, "error", error="Video download မအောင်မြင်ပါ")
                return
        else:
            set_status(job_id, "error", error="URL သို့မဟုတ် ဖိုင် လိုအပ်ပါတယ်")
            return

        # 2. Audio extract + Transcribe
        set_status(job_id, "audio", "🎧 Audio ထုတ်နေပါသည်...", 25)
        await asyncio.sleep(0.3)
        set_status(job_id, "transcribe", "🎤 စာသားပြောင်းနေပါသည် (Whisper)...", 40)
        transcript = transcribe(video_path)
        if not transcript:
            set_status(job_id, "error", error="စာသားထုတ်မရပါ")
            return

        # 3. Translate
        set_status(job_id, "translate", "🌐 မြန်မာလို ဘာသာပြန်နေပါသည်...", 55)
        translated = translate_text(transcript)

        # 4. TTS
        set_status(job_id, "tts", "🔊 အသံဖန်တီးနေပါသည် (TTS)...", 70)
        mp3_path = str(work / "tts.mp3")
        await tts(translated, mp3_path, voice, speed)
        audio_dur = get_duration(mp3_path) or 5.0

        # 5. SRT
        srt_path = None
        if mode == "auto":
            set_status(job_id, "srt", "📝 SRT subtitle ဖန်တီးနေပါသည်...", 80)
            srt_content = generate_srt(translated, audio_dur)
            srt_path = str(work / "subtitle.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

        # 6. Combine
        set_status(job_id, "combine", "🎞️ Video နဲ့ Audio ပေါင်းစပ်နေပါသည်...", 90)
        final_path = str(OUTPUT / f"{job_id}_final.mp4")
        ok = combine(video_path, mp3_path, final_path)
        if not ok:
            set_status(job_id, "error", error="Video ပေါင်းစပ်မရပါ")
            return

        audio_out = str(OUTPUT / f"{job_id}_audio.mp3")
        shutil.copy(mp3_path, audio_out)
        srt_out = None
        if srt_path:
            srt_out = str(OUTPUT / f"{job_id}_subtitle.srt")
            shutil.copy(srt_path, srt_out)

        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass

        set_status(job_id, "done", "✅ ပြီးပါပြီ!", 100, done=True, result={
            "audio_url": f"/download/{job_id}_audio.mp3",
            "video_url": f"/download/{job_id}_final.mp4",
            "srt_url": f"/download/{job_id}_subtitle.srt" if srt_out else None,
            "duration": round(audio_dur, 1),
            "voice": voice,
            "speed": speed,
            "mode": mode,
        })

    except Exception as e:
        set_status(job_id, "error", error=str(e)[:200])
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass

# ========== API ==========
@app.post("/api/start")
async def start_process(
    background_tasks: BackgroundTasks,
    url: str = Form(None),
    voice: str = Form("thiha"),
    speed: float = Form(1.4),
    mode: str = Form("auto"),
    file: UploadFile = File(None),
):
    job_id = uuid.uuid4().hex[:10]
    file_path = None

    if file and file.filename:
        work = TEMP / job_id
        work.mkdir(exist_ok=True)
        file_path = str(work / "input.mp4")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if os.path.getsize(file_path) < 1000:
            raise HTTPException(400, "Video file သေးလွန်းပါတယ်")
    elif not url or not url.strip():
        raise HTTPException(400, "URL သို့မဟုတ် Video ဖိုင် လိုအပ်ပါတယ်")

    set_status(job_id, "start", "⏳ စတင်နေပါသည်...", 5)
    background_tasks.add_task(run_job, job_id, (url or "").strip(), voice, speed, mode, file_path)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename)

@app.get("/api/health")
async def health():
    return {"status": "ok", "whisper": whisper_model is not None}