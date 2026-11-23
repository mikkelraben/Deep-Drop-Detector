"""Plot the per-epoch numbers that training.py prints, from a saved copy of its output.

Usage: python src/plot_metrics.py training_output.log [out.png]
       (default out.png: outputs/metrics.png)

One small panel per measure, all against epoch: train and validation loss,
share of validation drops warned, median seconds of warning, false alarms per
hour (with the MAX_FALSE_ALARMS limit) and the chosen P(no drop) threshold.
The dotted line marks the epoch that was saved (the one with the most drops
warned, like training.py picks it).
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from constants import ROOT_DIR

EPOCH_LINE = re.compile(
    r"Epoch (\d+), Loss: ([\d.]+), Val loss: ([\d.]+), Val: (\d+)% of drops warned, "
    r"median ([\d.]+) s ahead, ([\d.]+) false alarms per hour, threshold ([\d.e+-]+)")
TEST_LINE = re.compile(r"Test loss ([\d.]+), threshold ([\d.e+-]+): (\d+)% of drops warned, "
                       r"median ([\d.]+) s ahead, ([\d.]+) false alarms per hour")
MAX_FALSE_ALARMS = 15  # same limit as training.py

# Reference chart palette (light): categorical slots 1-2, text and surface tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893"


def parse(path):
    rows, test = [], None
    for line in open(path, encoding="utf-8"):
        if m := EPOCH_LINE.search(line):
            e, loss, val_loss, warned, ahead, fa, th = m.groups()
            rows.append(dict(epoch=int(e), loss=float(loss), val_loss=float(val_loss), warned=int(warned),
                             ahead=float(ahead), false_alarms=float(fa), threshold=float(th)))
        elif m := TEST_LINE.search(line):
            loss, th, warned, ahead, fa = m.groups()
            test = dict(loss=float(loss), threshold=float(th), warned=int(warned), ahead=float(ahead),
                        false_alarms=float(fa))
    if not rows:
        raise SystemExit(f"no 'Epoch ...' lines found in {path}")
    return rows, test


def saved_epoch(rows, test):
    """The epoch training.py saved: the one with the most drops warned. The log rounds to
    whole percents, so ties are broken by the threshold the test line reports (it is the
    saved epoch's threshold); without a test line, the first tied epoch is used."""
    top = max(r["warned"] for r in rows)
    tied = [r for r in rows if r["warned"] == top]
    if test:
        tied = [r for r in tied if r["threshold"] == test["threshold"]] or tied
    return tied[-1 if test else 0]["epoch"]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch", fontsize=8.5, color=INK_2)


def line(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)


def main():
    log = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else ROOT_DIR / "outputs" / "metrics.png"
    rows, test = parse(log)
    ep = [r["epoch"] for r in rows]
    best = saved_epoch(rows, test)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    (a_loss, a_warn, a_ahead), (a_fa, a_th, a_note) = axes

    style(a_loss, "Loss")
    line(a_loss, ep, [r["loss"] for r in rows], BLUE, "train")
    line(a_loss, ep, [r["val_loss"] for r in rows], ORANGE, "validation")
    a_loss.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    a_loss.set_ylim(bottom=0)

    style(a_warn, "Validation drops warned")
    line(a_warn, ep, [r["warned"] for r in rows], BLUE)
    a_warn.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    a_warn.set_ylim(0, 100)

    style(a_ahead, "Median warning time (s before the drop)")
    line(a_ahead, ep, [r["ahead"] for r in rows], BLUE)
    a_ahead.set_ylim(bottom=0)

    style(a_fa, "False alarms per hour (validation)")
    line(a_fa, ep, [r["false_alarms"] for r in rows], BLUE)
    a_fa.axhline(MAX_FALSE_ALARMS, color=MUTED, linewidth=1, linestyle="--")
    a_fa.text(ep[0], MAX_FALSE_ALARMS, f" limit {MAX_FALSE_ALARMS}/h", color=INK_2, fontsize=8, va="bottom")
    a_fa.set_ylim(0, max(MAX_FALSE_ALARMS, max(r["false_alarms"] for r in rows)) * 1.15)

    style(a_th, "Chosen threshold on P(no drop)")
    line(a_th, ep, [r["threshold"] for r in rows], BLUE)
    a_th.set_ylim(bottom=0)

    for ax in (a_loss, a_warn, a_ahead, a_fa, a_th):
        ax.axvline(best, color=MUTED, linewidth=1, linestyle=":")
    a_warn.text(best, 4, f" saved: epoch {best}", fontsize=8, color=INK_2, va="bottom")

    a_note.axis("off")
    b = rows[best - ep[0]]
    note = (f"Saved model: epoch {best}\n"
            f"  {b['warned']}% of drops warned\n"
            f"  median {b['ahead']:.1f} s ahead\n"
            f"  {b['false_alarms']:.1f} false alarms per hour\n"
            f"  threshold {b['threshold']:.2g}")
    if test:
        note += (f"\n\nTest songs, same model and threshold:\n"
                 f"  {test['warned']}% of drops warned\n"
                 f"  median {test['ahead']:.1f} s ahead\n"
                 f"  {test['false_alarms']:.1f} false alarms per hour\n"
                 f"  test loss {test['loss']:.4f}")
    a_note.text(0, 1, note, va="top", fontsize=9.5, color=INK, family="monospace", linespacing=1.5)

    fig.tight_layout(h_pad=2.5, w_pad=2)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
