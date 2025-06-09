import os
import random
import tempfile
import shutil
import requests
import cv2
import numpy as np
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import subprocess
import asyncio
import json
import uuid
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def split_prompt_sentences(prompt: str) -> list:
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', prompt) if s.strip()]

def generate_image(prompt: str, width: int = 720, height: int = 1280) -> np.ndarray:
    encoded_prompt = requests.utils.quote(prompt)
    seed = random.randint(1, 1_000_000)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed={seed}"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Image generation failed")
    image = np.array(Image.open(BytesIO(response.content)))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def generate_audio(text: str, lang: str = None, output_path: str = None) -> str:
    if lang is None:
        try:
            lang = detect(text)
        except:
            lang = 'fr'
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return output_path

def get_font(font_path="DejaVuSans-Bold.ttf", font_size=40):
    try:
        return ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[Font Load Error] {e}")
        return ImageFont.load_default()


def draw_text_dynamic(draw, frame_size, words, font, word_index, fade_duration_frames, current_frame, interval_words):
    w, h = frame_size
    y_text = h - 200
    start_index = max(0, word_index - interval_words + 1)
    text = " ".join(words[start_index:word_index + 1])
    text_width = draw.textlength(text, font=font)
    x_text = (w - text_width) // 2
    alpha = 255
    if current_frame % (fade_duration_frames * 2) > fade_duration_frames:
        alpha = int(255 * (1 - ((current_frame % fade_duration_frames) / fade_duration_frames)))
    draw.text((x_text, y_text), text, font=font, fill=(255, 255, 255, alpha), stroke_width=2, stroke_fill="black")

def create_video_segment(image: np.ndarray, audio_path: str, text: str, output_path: str):
    fps = 30
    audio = AudioSegment.from_mp3(audio_path)
    audio_duration = len(audio) / 1000.0
    total_frames = int(audio_duration * fps)
    h, w = image.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    words = text.split()
    word_display_interval = max(1, int(total_frames / (len(words) + 1)))
    fade_duration_frames = int(fps * 0.5)
    interval_words = random.randint(3, 4)

    font = get_font(font_size=40) 

    zoom_factor = 1.35
    zoom_step = (zoom_factor - 1.0) / total_frames

    for i in range(total_frames):
        current_zoom = 1.0 + zoom_step * i
        zoomed_frame = cv2.resize(image, None, fx=current_zoom, fy=current_zoom)
        zh, zw = zoomed_frame.shape[:2]
        crop_x = (zw - w) // 2
        crop_y = (zh - h) // 2
        frame = zoomed_frame[crop_y:crop_y + h, crop_x:crop_x + w]

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        current_word = min(len(words) - 1, i // word_display_interval)
        draw_text_dynamic(draw, (w, h), words, font, current_word, fade_duration_frames, i, interval_words)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()

    temp_output = output_path.replace('.mp4', '_temp.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', output_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', temp_output
    ], check=True)
    os.replace(temp_output, output_path)

def concatenate_videos(video_paths: list, output_path: str):
    list_file = os.path.join(os.path.dirname(output_path), 'file_list.txt')
    with open(list_file, 'w') as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_path], check=True)
    os.remove(list_file)

@app.get("/progress")
async def progress_endpoint(request: Request, prompt: str):
    uid = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"video_temp_{uid}")
    os.makedirs(temp_dir, exist_ok=True)
    video_output = os.path.join(temp_dir, "final_video.mp4")

    async def event_generator():
        sections = split_prompt_sentences(prompt)
        total_texts = len(sections)
        video_segments = []
        processed_count = 0

        try:
            for i, text in enumerate(sections):
                yield f"data: {json.dumps({'progress': int((processed_count / total_texts) * 100), 'status': 'processing'})}\n\n"
                img = generate_image(text)
                audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
                generate_audio(text, output_path=audio_path)
                segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
                create_video_segment(img, audio_path, text, segment_path)
                video_segments.append(segment_path)
                processed_count += 1
                yield f"data: {json.dumps({'progress': int((processed_count / total_texts) * 100), 'status': 'processing'})}\n\n"
                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'progress': 95, 'status': 'processing'})}\n\n"
            concatenate_videos(video_segments, video_output)

            while not os.path.exists(video_output) or os.path.getsize(video_output) < 1_000_000:
                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'progress': 100, 'status': 'done', 'video_id': uid})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/video_file")
async def get_video(video_id: str):
    folder_path = os.path.join(tempfile.gettempdir(), f"video_temp_{video_id}")
    video_path = os.path.join(folder_path, "final_video.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Vidéo introuvable")
    return FileResponse(video_path, media_type="video/mp4", filename="tiktok_video.mp4")

@app.delete("/cleanup")
async def cleanup_video(video_id: str):
    folder_path = os.path.join(tempfile.gettempdir(), f"video_temp_{video_id}")
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="Dossier introuvable")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
