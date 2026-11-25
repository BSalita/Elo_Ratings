"""Plot acbl_api_memory_drive.py logs: server memory per request, before and after.

    python scripts/plot_acbl_api_memory.py --before before.csv --after after.csv --out ACBL_API_MEMORY_drive.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=pathlib.Path, required=True)
    parser.add_argument("--after", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--limit-gb", type=float, default=64.0)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    for path, label, color in ((args.before, "old code", "#c0392b"), (args.after, "fixed", "#2471a3")):
        df = pl.read_csv(path).filter(pl.col("request") > 0)
        ax.plot(df["request"], df["anon_gb"], color=color, lw=1.6, label=f"{label}: held after request")
        if "peak_rss_gb" in df.columns:
            ax.plot(df["request"], df["peak_rss_gb"], color=color, lw=0.9, ls=":", label=f"{label}: peak during request")
    ax.axhline(args.limit_gb, color="#555555", lw=1, ls="--")
    ax.text(1, args.limit_gb + 0.6, f"container limit {args.limit_gb:.0f} GB", fontsize=8, color="#555555")
    ax.set_xlabel("request (alternating Club / Tournament, Players / Pairs)")
    ax.set_ylabel("API process memory, GB")
    ax.set_ylim(0, args.limit_gb + 6)
    ax.set_title("ACBL Elo API memory, both frames cached (prod's 64 GB mode)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
