"""Section-rank evaluation, the same measure as acbl_prediction_rank_eval.py.

A predictor is a per-board gap: NS pair strength minus EW pair strength
(the pair strength is the mean of the two seats). Each pair's prediction is
the mean of its boards' gaps (sign flipped for East-West); pairs with 8+
boards are ranked against the other pairs of the same session, section and
direction (fields of 4+ pairs) and compared with their actual session % by
Spearman. Fields where a predictor is constant score 0 (no information),
not NaN, so sparse features are not flattered.
"""
from __future__ import annotations

import numpy as np
import polars as pl

MIN_PAIR_BOARDS = 8
MIN_FIELD_PAIRS = 4
FIELD = ["session_id", "section_id", "direction"]


def seat_gap(n, s, e, w) -> np.ndarray:
    return ((np.asarray(n, np.float64) + np.asarray(s, np.float64)) / 2.0
            - (np.asarray(e, np.float64) + np.asarray(w, np.float64)) / 2.0)


def pair_table(ev: pl.DataFrame, gaps: dict[str, np.ndarray], mask: np.ndarray) -> pl.DataFrame:
    f = ev.select("session_id", "section_id", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS").with_columns(
        [pl.Series(k, v) for k, v in gaps.items()]
    ).filter(pl.Series(mask))
    names = list(gaps)
    parts = []
    for d, num, sign in (("NS", "Pair_Number_NS", 1.0), ("EW", "Pair_Number_EW", -1.0)):
        actual = pl.col("Pct_NS") if sign > 0 else 1.0 - pl.col("Pct_NS")
        parts.append(
            f.group_by("session_id", "section_id", num)
            .agg(actual.mean().alias("actual"), *[(sign * pl.col(k)).mean().alias(k) for k in names],
                 pl.len().alias("boards"))
            .filter(pl.col("boards") >= MIN_PAIR_BOARDS)
            .drop(num)
            .with_columns(pl.lit(d).alias("direction"))
        )
    return pl.concat(parts)


def field_spearman(pairs: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    g = (
        pairs.group_by(FIELD)
        .agg(pl.len().alias("pairs"),
             *[pl.corr(k, "actual", method="spearman").alias(k) for k in names])
        .filter(pl.col("pairs") >= MIN_FIELD_PAIRS)
    )
    return g.with_columns([pl.col(k).fill_nan(0.0).fill_null(0.0) for k in names]).sort(FIELD)


def summarize(fields: pl.DataFrame, names: list[str], ref: str | None = None) -> dict:
    out = {"fields": fields.height}
    for k in names:
        v = fields[k].to_numpy()
        row = {"spearman": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")}
        if ref and k != ref:
            d = v - fields[ref].to_numpy()
            row["vs_ref"] = float(d.mean())
            row["vs_ref_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        out[k] = row
    return out


def evaluate(ev: pl.DataFrame, gaps: dict[str, np.ndarray], sets: dict[str, np.ndarray], ref: str | None = None) -> dict:
    res = {}
    for name, mask in sets.items():
        pairs = pair_table(ev, gaps, mask)
        fields = field_spearman(pairs, list(gaps))
        res[name] = summarize(fields, list(gaps), ref)
    return res


def fit_line(gap: np.ndarray, pct: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    x, y = gap[mask], pct[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    slope = np.cov(x, y)[0, 1] / x.var(ddof=1)
    return float(y.mean() - slope * x.mean()), float(slope)


def fit_blend(gaps: list[np.ndarray], pct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    X = np.column_stack([g[mask] for g in gaps] + [np.ones(int(mask.sum()))])
    y = pct[mask]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    w, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return w[:-1]


def table(res: dict, names: list[str]) -> str:
    sets = list(res)
    head = "| predictor | " + " | ".join(f"{s} ({res[s]['fields']:,})" for s in sets) + " |"
    lines = [head, "|" + "---|" * (len(sets) + 1)]
    for k in names:
        cells = []
        for s in sets:
            r = res[s][k]
            c = f"{r['spearman']:.4f}"
            if "vs_ref" in r:
                c += f" ({r['vs_ref']:+.4f} ±{1.96*r['vs_ref_se']:.4f})"
            cells.append(c)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
