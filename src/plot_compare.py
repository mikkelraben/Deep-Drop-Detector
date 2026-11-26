"""Compare several training runs in one figure one IEEE column wide.

Usage: python src/plot_compare.py out.png "label=training_run4.log" "label=training_run7.log" ...
       (a .png also gets a .pdf next to it)

Top: loss per epoch, one colour per run: solid = validation, dashed = train.
Bottom: test drops warned and median warning time for each run. These come from
outputs/test_scores.csv (re-scored with the current false-alarm count) when the run's
log is listed there, else from the log's "Test ..." line.
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, PercentFormatter

from constants import ROOT_DIR
from plot_metrics import (BLUE, COLUMN_WIDTH, INK, INK_2, ORANGE, PAPER_RC, paper_line, paper_style, parse,
                          save)

COLORS = [BLUE, ORANGE, "#1f9e6e", "#8a5cd1"]  # one per run, in this order
SCORES = ROOT_DIR / "outputs" / "test_scores.csv"


def test_scores(log, test):
    """(share of drops warned in %, median warning time) on the test songs."""
    if SCORES.exists():
        for row in csv.DictReader(open(SCORES)):
            if (ROOT_DIR / row["log"]).resolve() == (ROOT_DIR / log).resolve():
                return float(row["warned"]) * 100, float(row["ahead"])
    if test is None:
        raise SystemExit(f"no test numbers for {log}")
    return test["warned"], test["ahead"]


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    if not specs or len(specs) > len(COLORS):
        raise SystemExit(f"give 1-{len(COLORS)} runs as label=log")
    runs = []
    for spec, color in zip(specs, COLORS):
        label, log = spec.split("=", 1)
        rows, test = parse(log)
        runs.append((label, rows, test_scores(log, test), color))

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(COLUMN_WIDTH, 3.5))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
        ax = fig.add_subplot(grid[0, :])
        bars = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

        paper_style(ax, "Loss")
        handles = []
        for label, rows, _, color in runs:
            ep = [r["epoch"] for r in rows]
            paper_line(ax, ep, [r["val_loss"] for r in rows], color)
            paper_line(ax, ep, [r["loss"] for r in rows], color, linestyle="--", alpha=0.7, markersize=0)
            handles.append(Line2D([], [], color=color, linewidth=1.2, label=label))
        handles += [Line2D([], [], color=INK_2, linewidth=1.2, marker="o", markersize=2.5, label="Validation"),
                    Line2D([], [], color=INK_2, linewidth=1.2, linestyle="--", label="Train")]
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Epoch", color=INK)
        ax.legend(handles=handles, frameon=False, loc="lower left", ncols=3, handlelength=1.4,
                  columnspacing=1.0, borderaxespad=0.2)

        xs = list(range(len(runs)))
        for b, (k, ylabel, fmt) in zip(bars, [(0, "Drops warned (test)", "{:.0f}%"),
                                              (1, "Median warning (test, s)", "{:.1f}")]):
            paper_style(b, ylabel)
            vals = [r[2][k] for r in runs]
            b.bar(xs, vals, width=0.65, color=[r[3] for r in runs], edgecolor="white", linewidth=1)
            for x, v in zip(xs, vals):
                b.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
            b.xaxis.set_major_locator(FixedLocator(xs))
            b.set_xticklabels([])  # the colours match the legend above
            b.tick_params(axis="x", length=0)
            b.set_ylim(0, max(vals) * 1.18)
        bars[0].set_ylim(0, 100)
        bars[0].yaxis.set_major_formatter(PercentFormatter(decimals=0))
        fig.align_ylabels([ax, bars[0]])
        fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.8)
        save(fig, out)


if __name__ == "__main__":
    main()
