"""Check mlBridgeAugmentLib's player Elo against production and against kernel.py.

1. Defaults reproduce the stored production tournament ratings.
2. merged + start ratings + K 6 / boost 6 / 200 boards equals kernel.py's replay.

  acbl-pipeline/.venv/Scripts/python.exe elo/platinum_bench/verify_library.py
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kernel  # noqa: E402
from common import load_rows  # noqa: E402

lib = kernel.lib


def compare(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[m] - b[m])
    print(f"{name}: rows {m.sum():,} of {len(a):,}  mean |diff| {d.mean():.6f}  max {d.max():.4f}  exact {(d < 1e-3).mean():.5f}")
    return d.max()


def main() -> None:
    df = load_rows("tournament")
    lay = kernel.Layout(df)
    out = lib._player_elo_frame(df, minimum_sessions=1, global_stdev_ns=lay.gsd_ns, global_stdev_ew=lay.gsd_ew)
    worst = 0.0
    for s in "NE":
        worst = max(worst, compare(f"defaults vs stored Before {s}", out[f"Elo_R_{s}_Before"], df[f"Elo_R_{s}_Before"]))
    # Start ratings from masterpoints only (same values both ways).
    starts = {s: (1455.0 + 10.0 * np.log1p(df[f"MasterPoints_{s}"].cast(pl.Float64).to_numpy())) for s in "NSEW"}
    df2 = df.with_columns([pl.Series(f"_start_{s}", starts[s]) for s in "NSEW"])
    kw = dict(k_base=6.0, boost=6.0, provisional_boost_until=200)
    out2 = lib._player_elo_frame(df2, minimum_sessions=1, global_stdev_ns=lay.gsd_ns, global_stdev_ew=lay.gsd_ew,
                                 merged=True, start_rating_cols=tuple(f"_start_{s}" for s in "NSEW"), **kw)
    # kernel.py init: first finite start value per slot.
    init = lay.first_values(True, {s: np.where(np.isfinite(starts[s]), starts[s], np.nan) for s in "NSEW"})
    # first_values takes the first row even when NaN; take the first finite instead.
    a, b, c, d, n_slots, _ = lay.slots[True]
    rank = np.empty(lay.n, dtype=np.int64)
    rank[lay.proc_order] = np.arange(lay.n)
    fr = pl.concat([pl.DataFrame({"slot": i, "rank": rank, "v": starts[s]}) for s, i in zip("NSEW", (a, b, c, d))])
    fr = fr.filter(pl.col("v").is_finite()).sort("rank").group_by("slot").first()
    init = np.full(n_slots, 1500.0)
    init[fr["slot"].to_numpy()] = fr["v"].to_numpy()
    r = kernel.replay(lay, merged=True, init=init, boost=6.0, boost_until=200, k_base=6.0)
    for s in "NSEW":
        worst = max(worst, compare(f"candidate library vs kernel.py {s}", out2[f"Elo_R_{s}_Before"], r[f"r_{s}"]))
    print("WORST", worst)


if __name__ == "__main__":
    main()
