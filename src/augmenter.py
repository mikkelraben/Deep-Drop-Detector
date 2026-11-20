import math
import pickle
import random

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from constants import *


def load_pickle(filepath: str) -> list:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def test_augmenter(
    aug: "Augmenter",
    songs_path: str = PKL,
    song: dict | None = None,
    ncols: int = 2,
):
    """Plots the original mel plus all available augmentations in a grid (drops in red)."""
    if song is None:
        songs = load_pickle(songs_path)
        song = next((s for s in songs if s["drops"]), songs[0])

    # Collect original + all supported augmentations
    panels = [(song, f"Original: {song['name']}")]
    for aug_name in aug.AUGMENTATIONS:
        if aug_name in ("speed_up", "speed_down", "pitch_up", "pitch_down", "compressed", "telephone"):
            if (song["name"], aug_name) not in aug.augmented:
                print(f"Skipping '{aug_name}' (not found in precomputed augmentations for {song['name']})")
                continue
        result = aug.augment(song, aug_name)
        panels.append((result, f"Augmented: {result['augmentation']}"))
        print(f"Song: {song['name']} | augmentation: {result['augmentation']} | shape: {result['mel'].shape}")

    total_panels = len(panels)
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 3.5 * nrows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for idx, (item, title) in enumerate(panels):
        ax = flat_axes[idx]
        img = librosa.display.specshow(
            item["mel"].T.astype(np.float32),
            sr=SR,
            hop_length=HOP,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        for t in item["drops"]:
            ax.axvline(x=t, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    # Hide any unused grid cells
    for ax in flat_axes[total_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


class Augmenter:
    AUGMENTATIONS = [
        "pitch_up", "pitch_down",
        "speed_up", "speed_down",
        "volume_up", "volume_down",
        "compressed", "telephone",
        "background_noise", "white_noise"
    ]

    def __init__(
        self,
        augmented_path: str = AUG_PKL,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # (track name, augmentation type) -> precomputed entry
        self.augmented = {
            (s["name"], s["augmentation"]): s for s in load_pickle(augmented_path)
        }

    def change_pitch(self, song: dict, direction: str | None = None) -> dict:
        """direction: 'up', 'down' or None (random)."""
        direction = direction or self.rng.choice(["up", "down"])
        return self.augmented[(song["name"], f"pitch_{direction}")]

    def change_speed(self, song: dict, direction: str | None = None) -> dict:
        """direction: 'up', 'down' or None (random)."""
        direction = direction or self.rng.choice(["up", "down"])
        return self.augmented[(song["name"], f"speed_{direction}")]

    def change_volume(self, song: dict, db_change: float = 0.0) -> dict:
        direction = "up" if db_change > 0 else "down"
        adjusted_mel = song["mel"].astype(np.float32) + db_change
        return {
            "name": song["name"],
            "mel": adjusted_mel.astype(song["mel"].dtype),
            "drops": song["drops"],
            "augmentation": f"volume_{direction}",
        }

    def add_compression(self, song: dict) -> dict:
        return self.augmented[(song["name"], "compressed")]

    def add_phone_filter(self, song: dict) -> dict:
        return self.augmented[(song["name"], "telephone")]

    def add_background_noise(self, song: dict) -> dict:
        return self.augmented[(song["name"], "background_noise")]

    def add_white_noise(self, song: dict, noise_level: float = 0.5) -> dict:
        mel = song["mel"].astype(np.float32)
        mel += self.np_rng.normal(0.0, noise_level, size=mel.shape)
        return {
            "name": song["name"],
            "mel": mel.astype(song["mel"].dtype),
            "drops": song["drops"],
            "augmentation": "white_noise",
        }

    def augment(self, song: dict, augmentation: str | None = None) -> dict:
        """
        Applies one augmentation to `song` ({'name', 'mel', 'drops', ...}).
        Random if `augmentation` is None; otherwise one of AUGMENTATIONS.
        """
        augmentation = augmentation or self.rng.choice(self.AUGMENTATIONS)

        match augmentation:
            case "pitch_up":
                return self.change_pitch(song, "up")
            case "pitch_down":
                return self.change_pitch(song, "down")
            case "speed_up":
                return self.change_speed(song, "up")
            case "speed_down":
                return self.change_speed(song, "down")
            case "volume_up":
                return self.change_volume(song, 6.0)
            case "volume_down":
                return self.change_volume(song, -6.0)
            case "compressed":
                return self.add_compression(song)
            case "telephone":
                return self.add_phone_filter(song)
            case "background_noise":
                return self.add_background_noise(song)
            case "white_noise":
                return self.add_white_noise(song)
            case _:
                raise ValueError(f"Unknown augmentation: {augmentation}")


if __name__ == "__main__":
    aug = Augmenter(seed=0)
    test_augmenter(aug, ncols=2)