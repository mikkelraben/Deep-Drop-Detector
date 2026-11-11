import logging
import random
import librosa
import numpy as np
from pathlib import Path

def add_white_noise(x, snr_db):
    """Add white Gaussian noise at a given signal-to-noise ratio (dB)."""
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    return x + np.random.randn(len(x)).astype(np.float32) * np.sqrt(noise_power)


def _fit_noise_length(noise, n):
    """Return exactly n samples: random crop if long enough, otherwise loop from a random offset."""
    if len(noise) >= n:
        start = random.randint(0, len(noise) - n)
        return noise[start:start + n]
    reps = int(np.ceil(n / len(noise))) + 1
    looped = np.tile(noise, reps)
    start = random.randint(0, len(noise) - 1)
    return looped[start:start + n]


def add_background_noise(x, noise, snr_db):
    """Mix a background noise recording into x at a given SNR (dB). `noise` is a 1D float array."""
    n = _fit_noise_length(noise, len(x))
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = np.mean(n ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    n = n * np.sqrt(target_noise_power / noise_power)
    return x + n.astype(np.float32)


def dynamic_range_compress(x, threshold_db=-20.0, ratio=4.0):
    """
    Crude static compressor: attenuate samples above the threshold by `ratio`.
    Sample-wise (no attack/release), so it adds a bit of distortion. Fine for a draft.
    """
    eps = 1e-8
    level_db = 20 * np.log10(np.abs(x) + eps)
    over = np.maximum(level_db - threshold_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    y = x * (10 ** (gain_db / 20))
    # makeup gain so the peak level roughly matches the input
    peak_in, peak_out = np.max(np.abs(x)) + eps, np.max(np.abs(y)) + eps
    return y * (peak_in / peak_out)


class WaveformAugment:
    """
    Random waveform augmentations. Each one fires with its own probability.

    Time stretch changes where the drop lands relative to the clip, so it is
    handled in two steps to keep the labels correct:
        1. rate = aug.sample_rate()       # dataset calls this BEFORE loading audio
        2. wav  = aug(wav, rate)          # wav holds clip_len * rate seconds of source audio,
                                          # and is stretched back to exactly clip_len seconds
    """
    def __init__(
        self,
        sr=44100,
        clip_len=30.0,
        p_stretch=0.3, stretch_range=(0.9, 1.1),
        p_pitch=0.3, semitone_range=(-2.0, 2.0),
        p_compress=0.3, threshold_db=(-30.0, -10.0), ratio=(2.0, 8.0),
        p_gain=0.5, gain_db=(-6.0, 6.0),
        p_white_noise=0.5, white_snr_db=(10.0, 30.0),
        bg_noise_dir=None, p_bg_noise=0.4, bg_snr_db=(5.0, 20.0),
        logger: logging.Logger | None=None
    ):
        self.logger = logger

        self.sr, self.clip_len = sr, clip_len
        self.p_stretch, self.stretch_range = p_stretch, stretch_range
        self.p_pitch, self.semitone_range = p_pitch, semitone_range
        self.p_compress, self.threshold_db, self.ratio = p_compress, threshold_db, ratio
        self.p_gain, self.gain_db = p_gain, gain_db
        self.p_white_noise, self.white_snr_db = p_white_noise, white_snr_db
        self.p_bg_noise, self.bg_snr_db = p_bg_noise, bg_snr_db

        # Load all background noise files once (mono, at the target sample rate)
        self.noise_bank = []
        if bg_noise_dir is not None:
            exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            for p in sorted(Path(bg_noise_dir).iterdir()):
                if p.suffix.lower() in exts:
                    y, _ = librosa.load(str(p), sr=sr, mono=True)
                    if len(y) > 0:
                        self.noise_bank.append(y.astype(np.float32))

    def sample_rate(self):
        """Speed factor: >1 = faster (so we need MORE source audio), <1 = slower."""
        if random.random() < self.p_stretch:
            return random.uniform(*self.stretch_range)
        return 1.0

    def __call__(self, wav, rate=1.0):
        x = wav.astype(np.float32)

        if rate != 1.0:
            x = librosa.effects.time_stretch(x, rate=rate)  # duration -> duration / rate

        if random.random() < self.p_pitch:
            n_steps = random.uniform(*self.semitone_range)
            x = librosa.effects.pitch_shift(x, sr=self.sr, n_steps=n_steps)

        if random.random() < self.p_compress:
            x = dynamic_range_compress(
                x, threshold_db=random.uniform(*self.threshold_db),
                ratio=random.uniform(*self.ratio),
            )

        if random.random() < self.p_gain:
            x = x * 10 ** (random.uniform(*self.gain_db) / 20)

        if random.random() < self.p_noise:
            x = add_white_noise(x, random.uniform(*self.snr_db))

        if self.noise_bank and random.random() < self.p_bg_noise:
            noise = random.choice(self.noise_bank)
            x = add_background_noise(x, noise, random.uniform(*self.bg_snr_db))

        # stretching can be off by a few samples -> force exact length
        n = int(round(self.clip_len * self.sr))
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))

        return np.clip(x, -1.0, 1.0)