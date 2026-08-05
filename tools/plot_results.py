#!/usr/bin/env python3
"""Render per-file boxplots (compression ratio, enc/dec speed) from results/<dataset>.csv.

The CSV rows are `file,compressor,originalSize,compressedSize,encSeconds,decSeconds`
as produced by running the benchmark binary per file (see README).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["lz4", "fsst", "zstd", "brotli", "tokenizer", "opp", "opp+lz4", "opp+zstd"]
MIB = float(1 << 20)


def draw(dataset: str, metric: str, ylabel: str, values: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    data = [values[c] for c in ORDER]
    boxes = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.55,
                       whis=(5, 95), medianprops={"color": "black"},
                       flierprops={"marker": ".", "markersize": 4, "alpha": 0.6})
    for patch, name in zip(boxes["boxes"], ORDER):
        patch.set_facecolor("#e8734a" if name == "opp" else "#9db8d2")
        patch.set_alpha(0.9)
    for pos, line in enumerate(boxes["medians"], start=1):
        median = line.get_ydata()[0]
        label = f"{median:.2f}" if median < 100 else f"{median:.0f}"
        ax.annotate(label, (pos + 0.31, median), va="center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset}: per-file {ylabel} ({len(data[0])} files)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path("plots") / f"{dataset}_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot(dataset: str) -> None:
    ratios, enc, dec = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(f"results/{dataset}.csv") as f:
        for row in csv.reader(f):
            compressor, original, compressed = row[1], int(row[2]), int(row[3])
            ratios[compressor].append(original / compressed)
            enc[compressor].append(original / MIB / float(row[4]))
            dec[compressor].append(original / MIB / float(row[5]))

    draw(dataset, "ratio", "compression ratio", ratios)
    draw(dataset, "enc", "encode speed [MiB/s]", enc)
    draw(dataset, "dec", "decode speed [MiB/s]", dec)


if __name__ == "__main__":
    for dataset in sys.argv[1:] or ["dbtext", "bitext"]:
        plot(dataset)
