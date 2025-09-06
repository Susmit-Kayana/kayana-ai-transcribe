import whisperx
import torch

# Automatic device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Download WhisperX models
print("Downloading WhisperX model...")
whisperx.load_model("base", device)

print("Downloading alignment model...")
_, _ = whisperx.load_align_model(language_code="en", device=device)

print("✅ Models downloaded for device:", device)
