import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from torch.utils.data import Dataset
from tqdm import tqdm

from augmentation import WaveformAugment


class DeepDropDataset(Dataset):
    """
    Each item: (features, label)
        features: stft(wav) if `stft` is given, else waveform tensor [1, clip_len * sr]
        label:    1.0 if a drop occurs in the `horizon` seconds right after the clip, else 0.0

    Args:
        tracks:     dict {track_name: {"drops": [{"t": seconds, ...}, ...], ...}}.
                    Use one split (train/val/test) per dataset, split BY TRACK to avoid leakage.
        audio_root: folder that the track names are relative to (original .m4a files).
        cache_dir:  where decoded mono wav files are stored. Tracks are converted once on first
                    use, then reused on every later run. Add this folder to .gitignore.
        clip_len:   length of the model input in seconds.
        horizon:    seconds after the clip in which a drop makes the label 1.
        sr:         sample rate. Must match WaveformAugment(sr=...).
        stride:     seconds between clip start times.
        augment:    WaveformAugment instance, or None (use None for val/test).
        stft:       callable waveform tensor [1, T] -> features, or None.
        logger:     optional logger for warnings (falls back to print).
    """

    def __init__(
        self,
        tracks: dict,
        audio_root: str | Path,
        cache_dir: str | Path = "assets/wav_cache",
        clip_len=30.0,
        horizon=5.0,
        sr=44100,
        stride=5.0,
        augment: WaveformAugment | None = None,
        stft=None,
        logger: logging.Logger | None = None,
    ):
        if augment is not None:
            assert augment.sr == sr, f"augment.sr ({augment.sr}) must match dataset sr ({sr})"

        self.logger = logger
        self.audio_root = Path(audio_root)
        # sr in the folder name so changing sr never reuses stale files
        self.cache_dir = Path(cache_dir) / f"sr{sr}"
        self.clip_len, self.horizon, self.sr, self.stride = clip_len, horizon, sr, stride
        self.augment = augment
        self.has_augment = augment is not None
        self.stft = stft
        self.has_stft = stft is not None

        self.drops = {
            name: sorted(d["t"] for d in info["drops"])
            for name, info in tracks.items()
        }

        # Longest source window we may ever need (clip + horizon, at max speed-up)
        self.max_rate = augment.stretch_range[1] if self.has_augment else 1.0
        self.max_rate = max(self.max_rate, 1.0)
        need = (clip_len + horizon) * self.max_rate

        # Convert missing tracks to wav (once), then build the (track, start_time) list
        self.samples = []
        self.skipped = []
        for name in tqdm(list(self.drops), desc="Building dataset"):
            try:
                path = self._ensure_wav(name)
                dur = librosa.get_duration(path=str(path))
            except (OSError, CouldntDecodeError) as e:
                # FileNotFoundError is an OSError. Most likely a filename ffmpeg can't handle
                # (unicode / special characters). Skip the track for now and fix the name later.
                self._warn(f"Skipping '{name}' ({type(e).__name__}) - probably a bad filename")
                self.skipped.append(name)
                continue

            for start in np.arange(0.0, dur - need, stride):
                self.samples.append((name, float(start)))

        # skipped tracks must not be looked up later
        for name in self.skipped:
            self.drops.pop(name, None)

        if self.skipped:
            self._warn(f"Skipped {len(self.skipped)} of {len(tracks)} tracks")

    def _warn(self, msg):
        if self.logger:
            self.logger.warning(msg)
        else:
            tqdm.write(f"Warning: {msg}")

    def _wav_path(self, name):
        return self.cache_dir / Path(name).with_suffix(".wav")

    def _ensure_wav(self, name):
        """Return the cached wav for a track, converting it once (mono, self.sr) if missing."""
        dst = self._wav_path(name)
        if dst.exists():
            return dst

        src = self.audio_root / name
        if not src.exists():
            raise FileNotFoundError(src)

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.wav")  # temp name so a crash never leaves a half-written wav
        try:
            audio = AudioSegment.from_file(str(src))
            audio = audio.set_channels(1).set_frame_rate(self.sr)  # stereo -> mono
            audio.export(str(tmp), format="wav")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dst)
        return dst

    def __len__(self):
        return len(self.samples)

    def _label(self, name, start, rate):
        """1 if any drop lies in the horizon window right after the clip, in SOURCE time."""
        src_clip = self.clip_len * rate
        src_horizon = self.horizon * rate
        lo, hi = start + src_clip, start + src_clip + src_horizon
        return float(any(lo <= t <= hi for t in self.drops[name]))

    def __getitem__(self, i):
        name, start = self.samples[i]

        # 1) decide stretch rate first: it determines how much audio to load and how to label
        rate = self.augment.sample_rate() if self.has_augment else 1.0

        # 2) load clip_len * rate seconds of source audio from the cached wav
        wav, _ = librosa.load(
            str(self._wav_path(name)), sr=self.sr, mono=True,
            offset=start, duration=self.clip_len * rate,
        )

        # 3) label against the scaled horizon (uses source time)
        label = self._label(name, start, rate)

        # 4) augment (stretch back to clip_len happens in here)
        if self.has_augment:
            wav = self.augment(wav, rate)
        else:
            n = int(round(self.clip_len * self.sr))
            wav = wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))

        wav = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]

        # 5) STFT / mel spectrogram
        feats = self.stft(wav) if self.has_stft else wav

        return feats, torch.tensor(label, dtype=torch.float32)