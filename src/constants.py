from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

PKL = SRC_DIR.parent / "data" / "songs_10fps.pkl"
AUG_PKL = SRC_DIR.parent / "data" / "songs_10fps_augmented.pkl"

SR = 22_050
FPS = 10
HOP = SR // FPS        # 2205 samples = 0.1 s per frame
SEED = 42