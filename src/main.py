"""
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O input.wav
"""
from faster_whisper import WhisperModel

model_id = "thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2"
device = "cpu"  # or "cuda"
compute_type = "int8"  # or "float16", "int8"
language = "he"  # Hebrew language code

model = WhisperModel(
    model_id,
    device=device,
    compute_type=compute_type,
    language=language,
)

segments, info = model.transcribe("input.wav", beam_size=5)

print("Detected language:", info.language, info.language_probability)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")