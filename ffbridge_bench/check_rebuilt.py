"""Check a rebuilt FFBridge Elo results parquet: which formula do its stored
ratings follow, and how do they rank club sessions?

  acbl-pipeline/.venv/Scripts/python.exe elo/ffbridge_bench/check_rebuilt.py <results.parquet>
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ffb_bench  # noqa: E402


def main() -> None:
    ffb_bench.RESULTS = pathlib.Path(sys.argv[1])
    data = ffb_bench.Data()
    d = data.d
    stored = (d["player1_scratch_elo_before"].cast(pl.Float64).to_numpy()
              + d["player2_scratch_elo_before"].cast(pl.Float64).to_numpy()) / 2
    old = data.replay()
    new = data.replay(k_base=24.0, boost=4.0, boost_n=10)
    m = np.isfinite(stored)
    for name, (o1, o2, _, _) in (("old formula (K 32)", old), ("new formula (K 24, 4x/10)", new)):
        pair = (o1 + o2) / 2
        rho = pl.DataFrame({"a": stored[m], "b": pair[m]}).select(pl.corr("a", "b", method="spearman")).item()
        print(f"stored vs replay of {name}: Spearman {rho:.4f}")
    year = d["Date"].dt.year().to_numpy()
    preds = {"old_formula": (old[0] + old[1]) / 2, "stored": np.nan_to_num(stored, nan=0.0)}
    g = ffb_bench.field_spearman(d, preds, year == 2026)
    print("2026 club sessions:", ffb_bench.summarize(g, list(preds), "old_formula"))
    print(f"rows {d.height:,}, last date {d['Date'].max()}")


if __name__ == "__main__":
    main()
