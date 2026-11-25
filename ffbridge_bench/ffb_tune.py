"""FFBridge K / provisional boost grid, chosen on 2025, reported on 2026.

  acbl-pipeline/.venv/Scripts/python.exe elo/ffbridge_bench/ffb_tune.py
"""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ffb_bench import WORK, Data, field_spearman, summarize  # noqa: E402


def main() -> None:
    started = datetime.now()
    data = Data()
    d = data.d
    year = d["Date"].dt.year().to_numpy()
    preds = {}
    o1, o2, _, _ = data.replay()
    preds["production"] = (o1 + o2) / 2
    for k, b, n, pe in itertools.product((20.0, 24.0, 32.0, 40.0), (2.0, 3.0, 4.0, 6.0), (10, 20, 40), (False, True)):
        o1, o2, _, _ = data.replay(k_base=k, boost=b, boost_n=n, pair_expect=pe)
        preds[f"k{k:g}_b{b:g}_n{n}{'_pair' if pe else ''}"] = (o1 + o2) / 2
    dev = summarize(field_spearman(d, preds, year == 2025), list(preds), "production")
    ranked = sorted(preds, key=lambda k: dev[k]["spearman"], reverse=True)
    best = ranked[0]
    test = summarize(field_spearman(d, {k: preds[k] for k in ("production", best, *ranked[1:5])}, year == 2026),
                     ["production", best, *ranked[1:5]], "production")
    print("dev 2025 top 10:")
    for k in ranked[:10]:
        r = dev[k]
        print(f"  {k}: {r['spearman']:.4f} ({r.get('vs_ref', 0):+.4f} ±{r.get('ci95', 0):.4f})")
    print(f"  production: {dev['production']['spearman']:.4f}")
    print(f"chosen on 2025: {best}; 2026 test:")
    for k in test:
        if k == "fields":
            continue
        r = test[k]
        print(f"  {k}: {r['spearman']:.4f} ({r.get('vs_ref', 0):+.4f} ±{r.get('ci95', 0):.4f})")
    (WORK / "results_tune.json").write_text(json.dumps({"dev": dev, "test": test, "chosen": best}, indent=1))
    print(f"elapsed {(datetime.now()-started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
