"""
uv run src/download.py
"""

import os
import io
import soundfile as sf
import pandas as pd
from datasets import load_dataset, Audio
from tqdm import tqdm

TARGET_DIR = "./dataset_output"
os.makedirs(TARGET_DIR, exist_ok=True)

ds = load_dataset("yanirmr/VoxKnesset", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

transcripts = pd.read_parquet("hf://datasets/yanirmr/voxknesset/transcripts.parquet")
text_lookup = dict(zip(transcripts["filename"], transcripts["text"]))

# We don't know the exact length of a streaming dataset easily, 
# but we can just iterate and show progress.
# If you want a specific number of files, you can use `take(N)`
# For now, we'll just download everything and show a counter.
print(f"Downloading files to {TARGET_DIR}...")

for sample in tqdm(ds, desc="Downloading audio files"):
    filename = os.path.basename(sample["audio"]["path"])
    filepath = os.path.join(TARGET_DIR, filename)
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        continue
        
    text = text_lookup.get(filename, "N/A")
    
    try:
        array, sampling_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]))
        sf.write(filepath, array, sampling_rate)
    except Exception as e:
        print(f"Error saving {filename}: {e}")

# Hard exit to prevent PyGILState_Release errors from huggingface/fsspec background threads during shutdown
os._exit(0)
