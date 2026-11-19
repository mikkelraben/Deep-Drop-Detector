"""Turn every labelled song into a 10 frames/s mel spectrogram (original + augmented)
and apply augmentations including background noise from a single background noise WAV file.

Usage: python build_features.py

songs_10fps.pkl holds a list with one dict per song:
    name:  track name
    mel:   [frames, 128] float16 log-mel (dB); frame k is centred at k / 10 s
    drops: drop times in seconds
    split: "train", "val" or "test" (70/15/15 by song, fixed seed)

songs_10fps_augmented.pkl holds a list with one dict per augmented track version:
    name:         track name
    mel:          [frames, 128] float16 log-mel (dB); frame k is centred at k / 10 s
    drops:        adjusted drop timestamps in seconds
    augmentation: type of augmentation applied (speed_up, speed_down, pitch_up, pitch_down, compressed, telephone, background_noise)
"""
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

# --- Configuration Constants ---
SR = 22_050
FPS = 10
HOP = SR // FPS        # 2205 samples = 0.1 s per frame
N_FFT = 4096           # longer than HOP, so no audio between frames is skipped
N_MELS = 128
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

# Augmentation default parameters
SPEED_FACTOR = 0.15   # 15% speed change (1.15x fast, 0.85x slow)
PITCH_STEPS = 1.5     # 1.5 semitones pitch change
NOISE_DB = -10.0      # Background noise level relative to song RMS in dB


def compress_audio(y):
    """Apply soft dynamic range compression."""
    c = 5.0
    return np.sign(y) * np.log1p(c * np.abs(y)) / np.log1p(c)


def telephone_filter(y, sr=SR, lowcut=300.0, highcut=3400.0):
    """Apply a telephone bandpass filter (combining high-pass and low-pass)."""
    sos = butter(4, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y)


def change_speed(y, drops, rate):
    """Stretch or compress audio speed and scale drop timestamps accordingly."""
    try:
        y_aug = librosa.effects.time_stretch(y, rate=rate, n_fft=N_FFT, hop_length=HOP)
    except Exception as e:
        if "ParameterError" in type(e).__name__:
            y_padded = np.pad(y, (0, SR), mode="constant")
            y_aug = librosa.effects.time_stretch(y_padded, rate=rate, n_fft=N_FFT, hop_length=HOP)
            y_aug = y_aug[: int(len(y) / rate)]
        else:
            raise e

    drops_aug = (np.array(drops) / rate).tolist()
    return y_aug, drops_aug


def add_background_noise(y, y_noise, noise_db=NOISE_DB):
    n_repeats = int(np.ceil(len(y) / len(y_noise)))
    y_noise = np.tile(y_noise, n_repeats)
    y_noise_segment = y_noise[:len(y)]
    
    # Match energy/RMS to achieve the desired noise_db relative to the song
    sig_rms = np.sqrt(np.mean(y**2)) + 1e-9
    noise_rms = np.sqrt(np.mean(y_noise_segment**2)) + 1e-9
    target_noise_rms = sig_rms * (10.0 ** (noise_db / 20.0))
    
    y_noise_scaled = y_noise_segment * (target_noise_rms / noise_rms)
    y_mixed = y + y_noise_scaled
    
    # Clip to prevent digital distortion/overflow
    return np.clip(y_mixed, -1.0, 1.0)


def augment_audio(y, drops, aug_type, y_noise=None, speed_delta=SPEED_FACTOR, pitch_delta=PITCH_STEPS):
    """Apply a specific augmentation to audio and adjust drop timestamps if necessary."""
    drops_aug = drops

    match aug_type:
        case "speed_up":
            y_aug, drops_aug = change_speed(y, drops, 1+speed_delta)
        case "speed_down":
            y_aug, drops_aug = change_speed(y, drops, 1-speed_delta)
        case "pitch_up":
            y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=pitch_delta)
        case "pitch_down":
            y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=-pitch_delta)
        case "compressed":
            y_aug = compress_audio(y)
        case "telephone":
            y_aug = telephone_filter(y, sr=SR)
        case "background_noise":
            y_aug = add_background_noise(y, y_noise)
        case _:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    return y_aug, drops_aug


def compute_mel(y):
    """Compute log-mel spectrogram (float16, [frames, 128])."""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).T.astype(np.float16)


def validate_drops(drops, length, key):
    """Check if any drops fall past the end of the song."""
    for t in drops:
        if t >= length:
            print(f"warning: drop at {t:.1f} s is past the end ({length:.1f} s) of {key}")


def main():
    tracks_path = Path("wav/drop_labels.json")
    if not tracks_path.exists():
        print(f"Metadata file {tracks_path} not found. Exiting.")
        return

    bg_noise_path = Path("bg_noise.wav")
    if not bg_noise_path.exists():
        print(f"Background noise file {bg_noise_path} not found. Exiting.")
        return
    y_bg_noise, _ = librosa.load(bg_noise_path, sr=SR, mono=True)

    with open(tracks_path, encoding="utf-8") as f:
        tracks = json.load(f)["tracks"]
    keys = sorted(k for k, tr in tracks.items() if tr["status"] in ("done", "no_drop"))

    order = np.random.default_rng(SEED).permutation(len(keys))
    n_train = round(SPLITS["train"] * len(keys))
    n_val = round(SPLITS["val"] * len(keys))
    
    split_of = {}
    for rank, i in enumerate(order):
        split_of[keys[i]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    songs_original = []
    songs_augmented = []
    augmentation_types = ["speed_up", "speed_down", "pitch_up", "pitch_down", "compressed", "telephone", "background_noise"]

    for i, key in enumerate(keys):
        wav_path = Path("wav") / f"{Path(key).stem}.wav"
        if not wav_path.exists():
            print(f"missing audio, skipped: {wav_path}")
            continue
        
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        base_drops = sorted(d["t"] for d in tracks[key]["drops"])
        mel_orig = compute_mel(y)
        length_orig = len(mel_orig) / FPS
        validate_drops(base_drops, length_orig, key)
        
        songs_original.append({
            "name": key, 
            "mel": mel_orig, 
            "drops": base_drops, 
            "split": split_of[key]
        })
        print(f"[{i + 1}/{len(keys)}] (Original) {split_of[key]:5s} {len(mel_orig):5d} frames  {key}")

        for aug_type in augmentation_types:
            y_aug, drops_aug = augment_audio(y, base_drops, aug_type, y_noise=y_bg_noise)
            mel_aug = compute_mel(y_aug)
            length_aug = len(mel_aug) / FPS
            validate_drops(drops_aug, length_aug, f"{key}_{aug_type}")
            
            songs_augmented.append({
                "name": key,
                "mel": mel_aug,
                "drops": drops_aug,
                "augmentation": aug_type
            })

    with open("songs_10fps.pkl", "wb") as f:
        pickle.dump(songs_original, f)
    counts = {s: sum(x["split"] == s for x in songs_original) for s in SPLITS}
    print(f"\nWrote songs_10fps.pkl: {len(songs_original)} songs {counts}")

    with open("songs_10fps_augmented.pkl", "wb") as f:
        pickle.dump(songs_augmented, f)
    print(f"Wrote songs_10fps_augmented.pkl: {len(songs_augmented)} augmented samples ({len(augmentation_types)} per track).")

if __name__ == "__main__":
    main()