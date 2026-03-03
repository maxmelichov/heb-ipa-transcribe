#!/usr/bin/env python3
"""
Audio IPA & Text Generator - MULTI GPU / MULTI WORKER
Monitoring Mode: Checks for new .wav files every X seconds.
uv run src/transcribe.py
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import librosa
import numpy as np
import soundfile as sf
from datetime import datetime
import multiprocessing
import queue
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Worker Function ---
def gpu_worker(worker_id: int, target_gpu: int, input_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    try:
        # Isolate the specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
        
        print(f"🚀 Worker {worker_id} starting on Physical GPU {target_gpu}...")
        
        ipa_model_id = "thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2"
        text_model_id = "ivrit-ai/whisper-large-v3-turbo-ct2"
        device = "cuda"
        compute_type = "int8"
        language = "he"
        max_chunk_s = 25
        sr = 16000

        # Initialize Models
        # device_index=0 because CUDA_VISIBLE_DEVICES restricts the view to only this GPU
        ipa_model = WhisperModel(ipa_model_id, device=device, device_index=0, compute_type=compute_type)
        text_model = WhisperModel(text_model_id, device=device, device_index=0, compute_type=compute_type)
        vad_model = load_silero_vad()
        
        print(f"✅ Worker {worker_id} Ready!")

        while True:
            task = input_queue.get() 
            if task is None: # Sentinel to stop
                break
            
            audio_path_str, relative_path = task
            
            try:
                # 1. Load Audio
                array, sampling_rate = sf.read(audio_path_str)
                if sampling_rate != sr:
                    array = librosa.resample(array, orig_sr=sampling_rate, target_sr=sr)
                audio = array.astype(np.float32)

                wav_tensor = torch.from_numpy(audio)
                
                # 2. VAD & Chunking
                timestamps = get_speech_timestamps(wav_tensor, vad_model, return_seconds=True, sampling_rate=sr)
                
                chunks = []
                for ts in timestamps:
                    chunk_start = int(ts["start"] * sr)
                    chunk_end = int(ts["end"] * sr)
                    max_samples = max_chunk_s * sr
                    while chunk_end - chunk_start > max_samples:
                        chunks.append((chunk_start, chunk_start + max_samples))
                        chunk_start += max_samples
                    if chunk_end > chunk_start:
                        chunks.append((chunk_start, chunk_end))

                if not chunks:
                    merged = []
                else:
                    merged = []
                    current_start, current_end = chunks[0]
                    for chunk_start, chunk_end in chunks[1:]:
                        if (chunk_end - current_start) <= max_chunk_s * sr:
                            current_end = chunk_end
                        else:
                            merged.append((current_start, current_end))
                            current_start, current_end = chunk_start, chunk_end
                    merged.append((current_start, current_end))

                # 3. Transcribe
                full_text = []
                full_ipa = []

                for chunk_start, chunk_end in merged:
                    chunk = audio[chunk_start:chunk_end]

                    text_segs, _ = text_model.transcribe(chunk, beam_size=5, language=language, temperature=0, condition_on_previous_text=False)
                    ipa_segs, _ = ipa_model.transcribe(chunk, beam_size=5, language=language, temperature=0, condition_on_previous_text=False, no_speech_threshold=0.1)

                    text_out = " ".join(s.text.strip() for s in text_segs).strip()
                    ipa_out = " ".join(s.text.strip() for s in ipa_segs).strip()

                    if text_out: full_text.append(text_out)
                    if ipa_out: full_ipa.append(ipa_out)

                # 4. Send Result
                result_queue.put({
                    'filename': relative_path,
                    'text': " ".join(full_text),
                    'phonemes': " ".join(full_ipa),
                    'processed_at': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing {relative_path}: {e}")
                result_queue.put({'error': str(e), 'filename': relative_path})

    except Exception as e:
        logger.critical(f"Worker {worker_id} crashed: {e}")
    finally:
        print(f"💤 Worker {worker_id} finished.")


class Transcriber:
    def __init__(self, input_dir: str, workers_per_gpu: int = 2):
        self.input_dir = Path(input_dir)
        self.output_csv = self.input_dir / "metadata_ipa.csv"
        self.checkpoint_file = self.input_dir / "checkpoint_ipa.json"
        
        # Detect GPUs
        try:
            self.num_gpus = torch.cuda.device_count()
        except Exception:
            self.num_gpus = 1
            
        if self.num_gpus == 0:
            logger.warning("No GPUs detected, falling back to CPU (will be slow)")
            self.num_gpus = 1
            
        self.workers_per_gpu = workers_per_gpu
        self.total_workers = self.num_gpus * self.workers_per_gpu
        self.save_interval = 20

    def load_checkpoint(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception: pass
        return {"processed": {}}

    def save_checkpoint(self, checkpoint: Dict):
        checkpoint["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def get_pending_tasks(self, processed_files: set) -> List[Tuple[str, str]]:
        """Scans directory for new .wav files not in checkpoint"""
        tasks = []
        all_wavs = list(self.input_dir.glob("**/*.wav"))
        for f in all_wavs:
            rel_path = str(f.relative_to(self.input_dir))
            if rel_path not in processed_files:
                tasks.append((str(f), rel_path))
        return tasks

    def run(self):
        """Processes all pending files and exits"""
        print(f"👀 Scanning {self.input_dir} for .wav files...")
        print(f"⚙️ Config: {self.num_gpus} GPUs, {self.workers_per_gpu} workers per GPU ({self.total_workers} total workers).")

        checkpoint = self.load_checkpoint()
        tasks = self.get_pending_tasks(set(checkpoint["processed"].keys()))

        if not tasks:
            print("✨ No new files to process. Exiting.")
            return

        print(f"\n📂 Found {len(tasks)} new files. Starting processing batch...")
        self.process_batch(tasks, checkpoint)
        print("✨ Processing complete.")

    def process_batch(self, tasks: List[Tuple[str, str]], checkpoint: Dict):
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        result_queue = manager.Queue()

        for t in tasks: input_queue.put(t)
        for _ in range(self.total_workers): input_queue.put(None)

        workers = []
        worker_id = 0
        for gpu_id in range(self.num_gpus):
            for _ in range(self.workers_per_gpu):
                p = multiprocessing.Process(
                    target=gpu_worker, 
                    args=(worker_id, gpu_id, input_queue, result_queue)
                )
                p.start()
                workers.append(p)
                worker_id += 1

        pbar = tqdm(total=len(tasks), desc="🔥 Processing", unit="file")
        completed = 0
        
        while completed < len(tasks):
            try:
                res = result_queue.get(timeout=5)
                if 'error' not in res:
                    checkpoint["processed"][res['filename']] = {
                        "filename": res['filename'],
                        "text": res.get('text', ''),
                        "phonemes": res.get('phonemes', '')
                    }
                
                completed += 1
                pbar.update(1)

                if completed % self.save_interval == 0:
                    self.save_checkpoint(checkpoint)
                    self.export_csv(checkpoint)
            except queue.Empty:
                if all(not p.is_alive() for p in workers): break

        pbar.close()
        for p in workers: p.join()
        
        self.save_checkpoint(checkpoint)
        self.export_csv(checkpoint)

    def export_csv(self, checkpoint):
        results = list(checkpoint["processed"].values())
        if results:
            # Handle old checkpoints that might not have text
            for r in results:
                if 'text' not in r: r['text'] = ''
            df = pd.DataFrame(results)[['filename', 'text', 'phonemes']]
            df.to_csv(self.output_csv, index=False)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    TARGET_DIR = "./dataset_output" 
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Defaults to 2 workers per GPU. Will use all available GPUs detected.
    generator = Transcriber(input_dir=TARGET_DIR, workers_per_gpu=2)
    
    try:
        generator.run()
    except KeyboardInterrupt:
        print("\n👋 Process stopped by user.")
