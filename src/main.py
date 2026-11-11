import json
import random
from torch.utils.data import DataLoader
from dataset import DeepDropDataset
from augmentation import WaveformAugment

def split_tracks(tracks: dict, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """Split by track so clips from one song never appear in multiple datasets.

    tracks: the loaded JSON dict (with a "tracks" key), or the tracks dict itself.
    Returns (train_tracks, val_tracks, test_tracks), each a dict {track_name: track_info}.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"

    tracks = tracks.get("tracks", tracks)
    names = sorted(tracks.keys())
    random.Random(seed).shuffle(names)

    n = len(names)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    n_train = n - n_val - n_test
    assert n_train > 0, "not enough tracks for this split"

    train_names = names[:n_train]
    val_names = names[n_train:n_train + n_val]
    test_names = names[n_train + n_val:]

    pick = lambda ns: {name: tracks[name] for name in ns}
    return pick(train_names), pick(val_names), pick(test_names)

def main():
    label_path = "assets\\drop2\\drop_labels.json"
    audio_root = "assets\\drop2\\Epic-Bass-Drops"
    noise_dir = "assets\\bg_noise"
    with open(label_path) as f:
        tracks = json.load(f)["tracks"]

    train_tracks, val_tracks, test_tracks = split_tracks(tracks)

    train_ds = DeepDropDataset(train_tracks, audio_root, augment=WaveformAugment(bg_noise_dir=noise_dir))  # augment on train
    val_ds = DeepDropDataset(val_tracks, audio_root)  # no augmentation on val
    test_ds = DeepDropDataset(test_tracks, audio_root)  # no augmentation on test

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    x, y = next(iter(train_loader))
    print(x.shape, y)

if __name__ == "__main__":
    main()