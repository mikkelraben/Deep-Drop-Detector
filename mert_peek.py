"""Run MERT over part of a song and plot what it outputs.

Usage: python mert_peek.py song.mp3 [start_s=0] [duration_s=60] [layer=12]
Writes mert_timeline.png and mert_similarity.png in the current directory.
"""
import sys

import librosa
import matplotlib
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "m-a-p/MERT-v1-95M"
SR = 24_000
CHUNK = 5 * SR      # MERT's pretraining context
FRAME = 320         # samples per MERT frame (75 Hz)

path = sys.argv[1]
start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
duration = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
layer = int(sys.argv[4]) if len(sys.argv) > 4 else 12   # 0 = input to layer 1, 12 = last

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, trust_remote_code=True)

# This transformers version returns hidden_states=None from the remote code,
# so capture the input of layer 0 and the output of every layer with hooks.
layers = model.encoder.layers
hs = []
layers[0].register_forward_pre_hook(
    lambda m, args, kwargs: hs.append((args[0] if args else kwargs["hidden_states"]).detach()),
    with_kwargs=True,
)
for lyr in layers:
    lyr.register_forward_hook(
        lambda m, args, output: hs.append((output[0] if isinstance(output, tuple) else output).detach())
    )

wav, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=duration)
print(f"audio: {len(wav) / SR:.1f} s from {start:.1f} s, layer {layer}")

feats, times = [], []
for c0 in range(0, len(wav), CHUNK):
    seg = wav[c0:c0 + CHUNK]
    if len(seg) < SR:                 # skip a leftover shorter than 1 s
        break
    inp = proc(seg, sampling_rate=SR, return_tensors="pt")
    hs.clear()
    with torch.inference_mode():
        model(**inp)
    f = hs[layer][0].numpy()          # [frames, 768]
    feats.append(f)
    times.append(start + (c0 + np.arange(len(f)) * FRAME + 200) / SR)

F = np.concatenate(feats)             # [T, 768]
t = np.concatenate(times)             # [T] seconds
print(f"features: {F.shape}  # [frames, 768], {len(F) / (t[-1] - t[0]):.1f} frames/s")

rms = librosa.feature.rms(y=wav)[0]
rt = start + librosa.times_like(rms, sr=SR)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 3]})
a1.plot(rt, rms)
a1.set_ylabel("loudness (RMS)")
v = np.percentile(np.abs(F), 99)
a2.imshow(F.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
          extent=[t[0], t[-1], F.shape[1], 0])
a2.set_ylabel(f"MERT dimension (layer {layer})")
a2.set_xlabel("time (s)")
a2.set_xlim(t[0], t[-1])
fig.tight_layout()
fig.savefig("mert_timeline.png", dpi=120)

# Self-similarity: how alike is every moment to every other moment?
Z = F - F.mean(0)                     # remove the offset shared by all frames
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
S = Z @ Z.T
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(S, cmap="viridis", vmin=-1, vmax=1, extent=[t[0], t[-1], t[-1], t[0]])
ax.set_xlabel("time (s)")
ax.set_ylabel("time (s)")
fig.colorbar(im, ax=ax, label="cosine similarity")
fig.tight_layout()
fig.savefig("mert_similarity.png", dpi=120)
print("wrote mert_timeline.png and mert_similarity.png")