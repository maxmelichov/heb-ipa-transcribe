"""
uv run src/transcribe.py
"""
import io
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import time
import multiprocessing

ipa_model_id = "thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2"
text_model_id = "ivrit-ai/whisper-large-v3-turbo-ct2"
compute_type = "int8"
language = "he"
max_chunk_s = 25
sr = 16000


def load_audio(path):
    array, sampling_rate = sf.read(path)
    if sampling_rate != sr:
        import librosa
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=sr)
    return array.astype(np.float32)


def get_chunks(audio):
    vad_model = load_silero_vad()
    wav_tensor = torch.from_numpy(audio)
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

    merged = []
    current_start, current_end = chunks[0]
    for chunk_start, chunk_end in chunks[1:]:
        if (chunk_end - current_start) <= max_chunk_s * sr:
            current_end = chunk_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = chunk_start, chunk_end
    merged.append((current_start, current_end))

    print(f"Chunks before merge: {len(chunks)} | after merge: {len(merged)}")
    print()
    return merged


def transcribe_worker(gpu_id, chunk_indices, audio, result_queue):
    device = f"cuda:{gpu_id}"
    text_model = WhisperModel(text_model_id, device=device, compute_type=compute_type)
    ipa_model = WhisperModel(ipa_model_id, device=device, compute_type=compute_type)

    for idx, (chunk_start, chunk_end) in chunk_indices:
        chunk = audio[chunk_start:chunk_end]
        offset = chunk_start / sr

        text_segs, _ = text_model.transcribe(chunk, beam_size=5, language=language, temperature=0, condition_on_previous_text=False)
        ipa_segs, _ = ipa_model.transcribe(chunk, beam_size=5, language=language, temperature=0, condition_on_previous_text=False, no_speech_threshold=0.1)

        text_out = " ".join(s.text.strip() for s in text_segs)
        ipa_out = " ".join(s.text.strip() for s in ipa_segs)

        result_queue.put((idx, offset, text_out, ipa_out))


def transcribe_all(audio, merged):
    num_gpus = torch.cuda.device_count() or 1

    # Distribute chunks across GPUs: [(original_index, (start, end)), ...]
    indexed_chunks = list(enumerate(merged))
    gpu_chunks = [indexed_chunks[i::num_gpus] for i in range(num_gpus)]

    result_queue = multiprocessing.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_chunks[gpu_id]:
            continue
        p = multiprocessing.Process(
            target=transcribe_worker,
            args=(gpu_id, gpu_chunks[gpu_id], audio, result_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect and sort by original chunk index
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda r: r[0])

    for _, offset, text_out, ipa_out in results:
        if text_out or ipa_out:
            print(f"[{offset:.2f}s]\t{text_out}\t{ipa_out}")


def main():
    audio = load_audio("433_2120_10152_10474.wav")

    start = time.time()

    merged = get_chunks(audio)
    transcribe_all(audio, merged)

    end = time.time()
    total_audio_s = len(audio) / sr
    elapsed_s = end - start
    rtf = elapsed_s / total_audio_s if total_audio_s > 0 else 0
    print(f"\nElapsed: {elapsed_s:.2f}s | Audio: {total_audio_s:.2f}s | RTF: {rtf:.3f}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
