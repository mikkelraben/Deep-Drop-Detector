"""Training windows for the drop detector, cut from songs_10fps.pkl (see build_features.py).

One sample is the `window` frames (default 300 = 30 s) up to and including frame
`end`, and its label says how soon the next drop comes after that last frame:
    0      no drop within the next 10 s
    1      drop 0-1 s ahead (including a drop on the last frame itself)
    2      drop 1-2 s ahead
    ...
    10     drop 9-10 s ahead
"""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from augmenter import Augmenter, load_pickle
from constants import *

WINDOW = 300                  # frames per sample (30 s)
BUCKET = FPS                  # 1 s per class
N_CLASSES = 11                # "no drop" + 10 one-second classes = a 10 s warning window
PAD_DB = -80.0                # silence, used before the song starts

def frame_labels(n_frames, drops):
    """Class for every possible window end frame 0..n_frames-1 of one song."""
    labels = np.zeros(n_frames, dtype=np.int64)
    # Later drops first, so a nearer drop overwrites a farther one.
    for d in sorted(drops, reverse=True):
        drop_frame = round(d * FPS)
        for ahead in range(BUCKET * (N_CLASSES - 1)):     # 0..99 frames = 0..9.9 s
            end = drop_frame - ahead
            if 0 <= end < n_frames:
                labels[end] = 1 + ahead // BUCKET
    return labels


class CustomDataset(Dataset):
    """Windows from the songs in one split ("train", "val" or "test").

    stride: take a window ending at every `stride`-th frame. Windows near a
    drop (labels 1-10) are always all kept, so a larger stride only thins out
    the "no drop" windows.
    augmentation: False for the original songs only; True to add one random
    augmented copy per song (fixed for the dataset's lifetime); "all" to add
    every precomputed augmented version (see pick_augmentations).
    window: frames per sample.
    """

    def __init__(self, split="train", stride=1, path=PKL, aug_path=AUG_PKL, seed=42, augmentation=False,
                 window=WINDOW):
        with open(path, "rb") as f:
            songs = [s for s in pickle.load(f) if s["split"] == split]
        if not songs:
            raise ValueError(f"no songs in split {split!r}")

        # Only songs of this split are augmented, so nothing leaks across splits.
        if augmentation == "all":
            # Every precomputed version of every song in this split, next to the originals.
            # Keyed by (name, augmentation) so duplicate entries in the pickle count once.
            names = {s["name"] for s in songs}
            extra = {(a["name"], a["augmentation"]): a for a in load_pickle(aug_path) if a["name"] in names}
            songs = songs + [extra[k] for k in sorted(extra)]
        elif augmentation:
            # One random augmented copy per song, added next to the originals.
            augmenter = Augmenter(augmented_path=aug_path, seed=seed)
            songs = songs + [augmenter.augment(s) for s in songs]

        self.window = window
        # Per song: which augmentation it is ("original" for none) and the index of its original.
        self.kinds = [s.get("augmentation", "original") for s in songs]
        first = {}
        for i, s in enumerate(songs):
            first.setdefault(s["name"], i)
        self.base = np.array([first[s["name"]] for s in songs])
        self.names = [s["name"] for s in songs]
        # Pad window - 1 silent frames in front, so drops near the start
        # still get full-length windows.
        pad = np.full((window - 1, songs[0]["mel"].shape[1]), PAD_DB, dtype=np.float16)
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
        # Padded row end + window - 1 is song frame `end`.
        x = self.mels[song][end:end + self.window].astype(np.float32)
        x = (x + 40.0) / 40.0                      # dB in [-80, 0] -> about [-1, 1]
        return torch.from_numpy(x), torch.tensor(self.labels[idx])

    def pick_augmentations(self, rng):
        """For a dataset built with augmentation="all": a mask over the windows that
        keeps every original song plus one randomly chosen augmented version of it.
        Call once per epoch with a numpy Generator for a fresh choice each time."""
        kinds = np.array(self.kinds)
        song_keep = kinds == "original"
        for b in np.unique(self.base):
            versions = np.flatnonzero((self.base == b) & ~song_keep)
            if len(versions):
                song_keep[rng.choice(versions)] = True
        return song_keep[self.index[:, 0]]

    def device_tensors(self, device):
        """The whole split on one device, for training without a DataLoader:
        (frames [all padded songs end to end, 128] float16, starts [n] first row of each
        window in frames, labels [n]). Cut windows out with windows(frames, starts)."""
        offsets = np.cumsum([0] + [len(m) for m in self.mels[:-1]])
        frames = torch.from_numpy(np.concatenate(self.mels)).to(device)
        starts = torch.from_numpy(offsets[self.index[:, 0]] + self.index[:, 1]).to(device)
        labels = torch.from_numpy(self.labels).to(device)
        return frames, starts, labels

    def class_weights(self):
        """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
        counts = np.bincount(self.labels, minlength=N_CLASSES).astype(np.float64)
        w = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32)

def windows(frames, starts, window=WINDOW):
    """Cut [len(starts), window, 128] normalized windows out of device_tensors' frames."""
    rows = starts[:, None] + torch.arange(window, device=starts.device)
    return (frames[rows].float() + 40.0) / 40.0   # dB in [-80, 0] -> about [-1, 1], like __getitem__


def augment(x, gain_db=6.0, noise_db=0.5, freq_mask=16, time_mask=20):
    """Cheap on-the-fly augmentation of a batch of normalized windows [B, T, F], on its device:
    a random loudness change of up to +-gain_db per window (like Augmenter's volume_up/down),
    a little white noise (like Augmenter's white_noise, noise_db std), and one band of up to
    freq_mask mel bins plus one stretch of up to time_mask frames set to the window's mean
    (SpecAugment-style masking)."""
    b, t, f = x.shape
    dev = x.device
    x = x + (torch.rand(b, 1, 1, device=dev) * 2 - 1) * (gain_db / 40.0)
    x = x + torch.randn_like(x) * (noise_db / 40.0)

    def band(size, max_width):
        width = torch.randint(0, max_width + 1, (b, 1), device=dev)
        start = (torch.rand(b, 1, device=dev) * (size - width + 1)).long()
        pos = torch.arange(size, device=dev)
        return (pos >= start) & (pos < start + width)   # [b, size]

    mask = band(f, freq_mask)[:, None, :] | band(t, time_mask)[:, :, None]
    return torch.where(mask, x.mean((1, 2), keepdim=True), x)


def test_augmentation():
    plain = CustomDataset(split="train", augmentation=False)
    aug = CustomDataset(split="train", augmentation=True)
    print(f"songs:   {len(plain.mels)} -> {len(aug.mels)}")
    print(f"windows: {len(plain)} -> {len(aug)} (x{len(aug) / len(plain):.2f})")


if __name__ == "__main__":
    test_augmentation()