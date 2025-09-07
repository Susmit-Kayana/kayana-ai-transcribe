import os
import json
import torch
import whisperx
import requests
from datetime import timedelta
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from dotenv import load_dotenv
from whisperx.diarize import assign_word_speakers
from pyannote.audio import Pipeline
import pandas as pd

# Suppress torchaudio backend warning
import warnings
warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.set_audio_backend has been deprecated"
)

# ========= Load Environment Variables =========
load_dotenv()

# ========= CONFIG =========
USE_OLLAMA = True  # True = Ollama, False = OpenAI
OLLAMA_MODEL = "llama3"  # local model name
HF_TOKEN = os.getenv("HF_TOKEN")  # HuggingFace token for diarization


# ========= Helpers =========
def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=round(seconds, 3))
    return str(td)[:-3].replace(".", ",")  # HH:MM:SS,mmm


def merge_segments(segments, max_gap: float = 0.6):
    merged = []
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]

        if merged and merged[-1]["speaker"] == speaker and (start - merged[-1]["end"]) < max_gap:
            merged[-1]["end"] = end
            merged[-1]["text"] += " " + text
        else:
            merged.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text
            })
    return merged


def rename_speakers(segments):
    speaker_map, new_segments, counter = {}, [], 1
    for seg in segments:
        old_speaker = seg.get("speaker", "Unknown")
        if old_speaker not in speaker_map:
            speaker_map[old_speaker] = f"Speaker {counter}"
            counter += 1
        new_segments.append({
            "speaker": speaker_map[old_speaker],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })
    return new_segments, speaker_map


def summarize_with_ollama(text: str, model: str = "llama3"):
    prompt = f"""
    You are an assistant that summarizes multi-speaker transcripts.
    Provide:
    1. A short summary (5 sentences max).
    2. Action items as bullet points.

    Transcript:
    {text}
    """
    response = requests.post("http://localhost:11434/api/generate",
                             json={"model": model, "prompt": prompt},
                             stream=False)
    return response.json()["response"]


# ========= FastAPI App =========
app = FastAPI()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Automatic device selection
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    elif torch.backends.mps.is_available():
        device = "mps"
        compute_type = "float32"  # FP16 not supported on MPS
    else:
        device = "cpu"
        compute_type = "float32"  # FP16 not supported on CPU

    print(f"Running on device: {device} with compute_type: {compute_type}")

    # Step 1: Load WhisperX model
    model = whisperx.load_model("base", device=device)
    if device == "cuda":
        model = model.half()  # optional half precision on GPU

    # Step 2: Transcribe
    asr_result = model.transcribe(
        temp_path,
        fp16=(compute_type == "float16")
    )

    # Step 3: Alignment
    align_model, metadata = whisperx.load_align_model(
        language_code=asr_result["language"], device=device
    )
    asr_aligned = whisperx.align(
        asr_result["segments"], align_model, metadata, temp_path, device
    )

    # Step 4: Diarization using pyannote.audio
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN
    )
    diarization_result = diarization_pipeline(temp_path)

    # Convert diarization result to DataFrame
    diarize_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        diarize_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    diarize_df = pd.DataFrame(diarize_segments)

    # Step 5: Assign word-level speakers
    friendly_segments, word_segments = assign_word_speakers(
        diarize_df, asr_aligned["segments"]
    )

    # Step 6: Rename speakers
    renamed_segments, mapping = rename_speakers(friendly_segments)

    # Step 7: Export JSON
    with open("transcript.json", "w", encoding="utf-8") as f:
        json.dump(renamed_segments, f, indent=2, ensure_ascii=False)

    # Step 8: Export SRT
    with open("transcript.srt", "w", encoding="utf-8") as f:
        for i, seg in enumerate(renamed_segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    # Step 9: Export VTT
    with open("transcript.vtt", "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in renamed_segments:
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    # Step 10: Summarization (optional)
    # full_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in renamed_segments)
    # summary = summarize_with_ollama(full_text, model=OLLAMA_MODEL) if USE_OLLAMA else ""
    # with open("summary.txt", "w", encoding="utf-8") as f:
    #     f.write(summary)

    # Cleanup temp
    os.remove(temp_path)

    return JSONResponse({
        "mapping": mapping,
        "files": {
            "json": "transcript.json",
            "srt": "transcript.srt",
            "vtt": "transcript.vtt",
            # "summary": "summary.txt"
        }
    })


@app.get("/download/{file_type}")
async def download_file(file_type: str):
    if file_type not in ["json", "srt", "vtt", "summary"]:
        return {"error": "Invalid file type"}
    filename = f"transcript.{file_type}" if file_type != "summary" else "summary.txt"
    return FileResponse(filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
