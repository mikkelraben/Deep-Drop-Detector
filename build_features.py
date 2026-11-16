"""Turn every labelled song into a 10 frames/s mel spectrogram and save them all.

Usage: python build_features.py   (reads wav/drop_labels.json, writes songs_10fps.pkl)

songs_10fps.pkl holds a list with one dict per song:
    name:  track name
    mel:   [frames, 128] float16 log-mel (dB); frame k is centred at k / 10 s
    drops: drop times in seconds
    split: "train", "val" or "test" (70/15/15 by song, fixed seed)
"""
import json
import pickle
from pathlib import Path

import librosa
import numpy as np

SR = 22_050
FPS = 10
HOP = SR // FPS        # 2205 samples = 0.1 s per frame
N_FFT = 4096           # longer than HOP, so no audio between frames is skipped
N_MELS = 128
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

tracks = json.load(open("wav/drop_labels.json"))["tracks"]
keys = sorted(k for k, tr in tracks.items() if tr["status"] in ("done", "no_drop"))

# Split by song, so overlapping windows from one song never land in two sets.
order = np.random.default_rng(SEED).permutation(len(keys))
n_train = round(SPLITS["train"] * len(keys))
n_val = round(SPLITS["val"] * len(keys))
split_of = {}
for rank, i in enumerate(order):
    split_of[keys[i]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

songs = []
for i, key in enumerate(keys):
    wav = Path("wav") / f"{Path(key).stem}.wav"
    if not wav.exists():
        print(f"missing audio, skipped: {wav}")
        continue
    y, _ = librosa.load(wav, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float16)   # [frames, 128]
    drops = sorted(d["t"] for d in tracks[key]["drops"])
    length = len(mel) / FPS
    for t in drops:
        if t >= length:
            print(f"warning: drop at {t:.1f} s is past the end ({length:.1f} s) of {key}")
    songs.append({"name": key, "mel": mel, "drops": drops, "split": split_of[key]})
    print(f"[{i + 1}/{len(keys)}] {split_of[key]:5s} {len(mel):5d} frames  {key}")

with open("songs_10fps.pkl", "wb") as f:
    pickle.dump(songs, f)
counts = {s: sum(x["split"] == s for x in songs) for s in SPLITS}
print(f"wrote songs_10fps.pkl: {len(songs)} songs {counts}")
