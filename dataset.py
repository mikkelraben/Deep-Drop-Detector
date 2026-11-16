"""Training windows for the drop detector, cut from songs_10fps.pkl (see build_features.py).

One sample is the 300 frames (30 s) up to and including frame `end`, and its
label says how soon the next drop comes after that last frame:
    0      no drop within the next 5 s
    1      drop 0-0.5 s ahead (including a drop on the last frame itself)
    2      drop 0.5-1 s ahead
    ...
    10     drop 4.5-5 s ahead
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

FPS = 10                      # frames per second in songs_10fps.pkl
WINDOW = 300                  # frames per sample (30 s)
BUCKET = FPS // 2             # 0.5 s per class
N_CLASSES = 11
PAD_DB = -80.0                # silence, used before the song starts
PKL = os.path.join(os.path.dirname(__file__), "songs_10fps.pkl")


def frame_labels(n_frames, drops):
    """Class for every possible window end frame 0..n_frames-1 of one song."""
    labels = np.zeros(n_frames, dtype=np.int64)
    # Later drops first, so a nearer drop overwrites a farther one.
    for d in sorted(drops, reverse=True):
        drop_frame = round(d * FPS)
        for ahead in range(BUCKET * (N_CLASSES - 1)):     # 0..49 frames = 0..4.9 s
            end = drop_frame - ahead
            if 0 <= end < n_frames:
                labels[end] = 1 + ahead // BUCKET
    return labels


class CustomDataset(Dataset):
    """Windows from the songs in one split ("train", "val" or "test").

    stride: take a window ending at every `stride`-th frame. Windows near a
    drop (labels 1-10) are always all kept, so a larger stride only thins out
    the "no drop" windows.
    """

    def __init__(self, split="train", stride=1, path=PKL):
        with open(path, "rb") as f:
            songs = [s for s in pickle.load(f) if s["split"] == split]
        if not songs:
            raise ValueError(f"no songs in split {split!r}")

        self.names = [s["name"] for s in songs]
        # Pad WINDOW - 1 silent frames in front, so drops in the first 30 s
        # still get full-length windows.
        pad = np.full((WINDOW - 1, songs[0]["mel"].shape[1]), PAD_DB, dtype=np.float16)
        self.mels = [np.concatenate([pad, s["mel"]]) for s in songs]

        index, labels = [], []
        for i, s in enumerate(songs):
            lab = frame_labels(len(s["mel"]), s["drops"])
            ends = np.arange(len(lab))
            keep = (ends % stride == 0) | (lab > 0)
            index.append(np.stack([np.full(keep.sum(), i), ends[keep]], axis=1))
            labels.append(lab[keep])
        self.index = np.concatenate(index)        # [n, 2]: (song, end frame)
        self.labels = np.concatenate(labels)      # [n]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        song, end = self.index[idx]
        # Padded row end + WINDOW - 1 is song frame `end`.
        x = self.mels[song][end:end + WINDOW].astype(np.float32)
        x = (x + 40.0) / 40.0                      # dB in [-80, 0] -> about [-1, 1]
        return torch.from_numpy(x), torch.tensor(self.labels[idx])

    def class_weights(self):
        """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
        counts = np.bincount(self.labels, minlength=N_CLASSES).astype(np.float64)
        w = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32)
