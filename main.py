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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= UTILS =========

def split_prompt_single(prompt: str) -> list:
    return [prompt.strip()]

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

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def get_line_height(draw, font):
    bbox = draw.textbbox((0, 0), "Ay", font=font)
    return bbox[3] - bbox[1] + 10

def get_font_for_text(draw, text, max_width, base_font_path="arial.ttf", font_size=200):
    try:
        font = ImageFont.truetype(base_font_path, font_size)
    except:
        font = ImageFont.load_default()
    lines = wrap_text(draw, text, font, max_width)
    return font, lines

def create_video_segment(image: np.ndarray, audio_path: str, text: str, output_path: str):
    fps = 30
    audio = AudioSegment.from_mp3(audio_path)
    audio_duration = len(audio) / 1000.0
    total_frames = int(audio_duration * fps)
    h, w = image.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame_idx in range(total_frames):
        current_time = frame_idx / fps
        zoom_factor = 1.0 + (0.1 * (current_time / audio_duration))
        zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
        zh, zw = zoomed.shape[:2]
        x1, y1 = (zw - w) // 2, (zh - h) // 2
        zoomed = zoomed[y1:y1+h, x1:x1+w]

        pil_img = Image.fromarray(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        font, lines = get_font_for_text(draw, text, max_width=w - 100)
        line_height = get_line_height(draw, font)
        total_text_height = len(lines) * line_height
        y_text = h - total_text_height - 100

        margin = 30
        rect_top = y_text - margin
        rect_bottom = y_text + total_text_height + margin
        draw.rectangle([(0, rect_top), (w, rect_bottom)], fill=(0, 0, 0, int(255 * 0.6)))

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_text = (w - text_width) // 2
            draw.text((x_text, y_text), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
            y_text += line_height

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

# ========= ENDPOINTS =========

@app.get("/progress")
async def progress_endpoint(request: Request, prompt: str):
    uid = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"video_temp_{uid}")
    os.makedirs(temp_dir, exist_ok=True)
    video_output = os.path.join(temp_dir, "final_video.mp4")

    async def event_generator():
        sections = split_prompt_single(prompt)
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

            # Attendre que le fichier soit pleinement écrit
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
