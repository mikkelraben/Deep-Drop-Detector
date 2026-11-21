"""Turn audio into a spectrum sampled 10 times a second with 100 Hz wide bins."""
import numpy as np


def sample_spectrum(y, sr, frame_rate=10, freq_res=100):
    """Power spectrum of `y`, `frame_rate` frames per second, one bin per `freq_res` Hz.

    A 100 Hz resolution needs a window of sr / 100 samples (10 ms). A frame
    every 100 ms would then skip 90 ms of audio between windows, so instead
    each frame averages the power of all half-overlapping 10 ms Hann windows
    whose centre lies in its 100 ms block (Welch's method). Every sample
    counts, and the averaging also makes the spectrum less noisy.

    Args:
        y: mono audio, 1-D array.
        sr: sample rate in Hz; must be divisible by `freq_res` and `frame_rate`
            (e.g. load with librosa.load(path, sr=24_000)).
        frame_rate: frames per second.
        freq_res: spacing between frequency bins in Hz.

    Returns:
        S: [n_frames, n_bins] power (use librosa.power_to_db(S) for dB).
        times: [n_frames] centre of each frame in seconds.
        freqs: [n_bins] bin frequencies in Hz: 0, 100, 200, ..., sr / 2.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr % freq_res or sr % frame_rate:
        raise ValueError(f"sr={sr} must be divisible by {freq_res} and {frame_rate}")

    n_win = sr // freq_res           # window length giving freq_res Hz bins
    hop = sr // frame_rate           # samples per output frame
    step = max(n_win // 2, 1)        # 50 % overlap between windows
    n_frames = -(-len(y) // hop)     # ceil: a partial last block still gets a frame

    # Centre window j at sample j * step; pad so edge windows fit.
    centres = np.arange(0, n_frames * hop, step)
    pad = n_win // 2
    yp = np.pad(y, (pad, n_frames * hop - len(y) + n_win))
    windows = np.lib.stride_tricks.sliding_window_view(yp, n_win)[centres]

    win = np.hanning(n_win + 2)[1:-1]  # Hann without its zero end points
    power = np.abs(np.fft.rfft(windows * win, axis=1)) ** 2 / np.sum(win ** 2)

    # Average the windows that belong to each frame.
    block = centres // hop
    S = np.zeros((n_frames, power.shape[1]))
    np.add.at(S, block, power)
    S /= np.bincount(block, minlength=n_frames)[:, None]

    times = (np.arange(n_frames) + 0.5) * hop / sr
    freqs = np.fft.rfftfreq(n_win, 1 / sr)
    return S, times, freqs
