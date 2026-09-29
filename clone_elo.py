"""Clone Elo: the rating a player would have partnered with a copy of themselves.

A pair rating is ``R0 + c_A + c_B``. ``Clone_Elo(A) = R0 + 2·c_A``. Contributions
come from a ridge regression over every pair session, so a partner's strength is
measured by how that partner does with everyone else.

``k_scale`` defaults to the classic ``400/ln(10)`` points per natural-log odds.
When the session rows carry current Elo, the default instead estimates that
factor from those ratings so ``Clone_Elo`` and ``Partner_Effect`` share the
stored Elo scale (FFBridge's chess-standardized columns, or ACBL's ``Elo_R_*``).
"""

from __future__ import annotations

import math
import threading
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl

K_SCALE_400 = 400.0 / math.log(10)
DEFAULT_R0 = 1500.0
DEFAULT_TAU_DAYS = 365.0
DEFAULT_RIDGE_LAMBDA = 10.0
DEFAULT_MIN_SESSIONS = 20
DEFAULT_MIN_PARTNERS = 3
_PCT_LO = 0.5
_PCT_HI = 99.5
_CALIBRATE_MIN_ROWS = 30

CLONE_COLUMNS = (
    "player_id",
    "Clone_Elo",
    "Clone_Pct",
    "Clone_SD",
    "Clone_N",
    "Clone_Partners",
    "Partner_Effect",
    "Clone_Status",
)

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple, pl.DataFrame] = {}
_CACHE_MAX = 8


def empty_clone_ratings() -> pl.DataFrame:
    """Zero-row frame with the clone-rating schema. Safe to register in DuckDB."""
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "Clone_Elo": pl.Int64,
            "Clone_Pct": pl.Float64,
            "Clone_SD": pl.Float64,
            "Clone_N": pl.Int64,
            "Clone_Partners": pl.Int64,
            "Partner_Effect": pl.Int64,
            "Clone_Status": pl.Utf8,
        }
    )


def register_clone_ratings(con: Any, ratings: pl.DataFrame) -> None:
    """Register ``clone_ratings`` for leaderboard SQL. Replaces a previous registration."""
    try:
        con.unregister("clone_ratings")
    except Exception:
        pass
    frame = ratings if ratings is not None and not ratings.is_empty() else empty_clone_ratings()
    if frame.is_empty() and "player_id" not in frame.columns:
        frame = empty_clone_ratings()
    con.register("clone_ratings", frame)


def logistic(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit_pct(pct: np.ndarray) -> np.ndarray:
    clipped = np.clip(pct, _PCT_LO, _PCT_HI) / 100.0
    return np.log(clipped / (1.0 - clipped))


def _player_id_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    lowered = text.str.to_lowercase()
    return (
        pl.when(
            text.is_null()
            | (text == "")
            | lowered.is_in(["none", "nan", "null", "<na>"])
        )
        .then(pl.lit(""))
        .otherwise(text.str.replace(r"\.0$", ""))
        .alias(column)
    )


def _require_score(score: str) -> str:
    if score not in ("Scratch", "Handicap"):
        raise ValueError(f"score must be 'Scratch' or 'Handicap', got {score!r}")
    return score


def _sessions_from_ffbridge(sessions_df: pl.DataFrame, score: str) -> pl.DataFrame:
    kind = "handicap" if score == "Handicap" else "scratch"
    pct_col = f"National_{score}_Pct"
    if pct_col not in sessions_df.columns:
        raise ValueError(
            f"FFBridge clone ratings need {pct_col}. Club percentages are not mixed in."
        )
    elo_a = f"player1_{kind}_elo_after"
    elo_b = f"player2_{kind}_elo_after"
    if "date" not in sessions_df.columns:
        raise ValueError("FFBridge clone ratings need a date column")
    if "player1_id" not in sessions_df.columns or "player2_id" not in sessions_df.columns:
        raise ValueError("FFBridge clone ratings need player1_id and player2_id")
    elo_a_expr = (
        pl.col(elo_a).cast(pl.Float64, strict=False)
        if elo_a in sessions_df.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    elo_b_expr = (
        pl.col(elo_b).cast(pl.Float64, strict=False)
        if elo_b in sessions_df.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    return sessions_df.select(
        _player_id_expr("player1_id").alias("player_a"),
        _player_id_expr("player2_id").alias("player_b"),
        pl.col("date").cast(pl.Date, strict=False).alias("date"),
        pl.col(pct_col).cast(pl.Float64, strict=False).alias("pct"),
        elo_a_expr.alias("elo_a"),
        elo_b_expr.alias("elo_b"),
    )


def _sessions_from_acbl_boards(boards: pl.DataFrame) -> pl.DataFrame:
    required = {"Player_ID_N", "Player_ID_S", "Player_ID_E", "Player_ID_W", "Pct_NS", "Date", "session_id"}
    missing = sorted(required - set(boards.columns))
    if missing:
        raise ValueError(f"ACBL clone ratings need columns {missing}")

    def _side(a: str, b: str, elo_a: str, elo_b: str, pct_expr: pl.Expr) -> pl.DataFrame:
        elo_a_expr = (
            pl.col(elo_a).cast(pl.Float64, strict=False)
            if elo_a in boards.columns
            else pl.lit(None, dtype=pl.Float64)
        )
        elo_b_expr = (
            pl.col(elo_b).cast(pl.Float64, strict=False)
            if elo_b in boards.columns
            else pl.lit(None, dtype=pl.Float64)
        )
        return (
            boards.select(
                _player_id_expr(a).alias("player_a"),
                _player_id_expr(b).alias("player_b"),
                pl.col("session_id").cast(pl.Utf8),
                pl.col("Date").cast(pl.Date, strict=False).alias("date"),
                pct_expr.alias("pct"),
                elo_a_expr.alias("elo_a"),
                elo_b_expr.alias("elo_b"),
            )
            .filter(pl.col("pct").is_not_null() & pl.col("pct").is_between(0.0, 100.0))
            .group_by(["player_a", "player_b", "session_id", "date"])
            .agg(
                pl.col("pct").mean(),
                pl.col("elo_a").mean(),
                pl.col("elo_b").mean(),
            )
            .drop("session_id")
        )

    pct = pl.col("Pct_NS").cast(pl.Float64, strict=False)
    ns = _side("Player_ID_N", "Player_ID_S", "Elo_R_N", "Elo_R_S", pct * 100.0)
    ew = _side("Player_ID_E", "Player_ID_W", "Elo_R_E", "Elo_R_W", (1.0 - pct) * 100.0)
    return pl.concat([ns, ew], how="vertical")


def _canonical_sessions(sessions_df: pl.DataFrame, score: str) -> pl.DataFrame:
    columns = set(sessions_df.columns)
    if {"player_a", "player_b", "pct"}.issubset(columns):
        date_col = "date" if "date" in columns else "Date" if "Date" in columns else None
        if date_col is None:
            raise ValueError("Clone sessions need a date or Date column")
        elo_a = pl.col("elo_a").cast(pl.Float64, strict=False) if "elo_a" in columns else pl.lit(None, dtype=pl.Float64)
        elo_b = pl.col("elo_b").cast(pl.Float64, strict=False) if "elo_b" in columns else pl.lit(None, dtype=pl.Float64)
        return sessions_df.select(
            _player_id_expr("player_a"),
            _player_id_expr("player_b"),
            pl.col(date_col).cast(pl.Date, strict=False).alias("date"),
            pl.col("pct").cast(pl.Float64, strict=False).alias("pct"),
            elo_a.alias("elo_a"),
            elo_b.alias("elo_b"),
        )
    if "player1_id" in columns:
        return _sessions_from_ffbridge(sessions_df, score)
    if "Player_ID_N" in columns and "Pct_NS" in columns:
        if score != "Scratch":
            raise ValueError(
                "ACBL clone ratings use matchpoint percentage; score must be 'Scratch'"
            )
        return _sessions_from_acbl_boards(sessions_df)
    raise ValueError(
        "sessions_df must be FFBridge pair sessions (player1_id, National_*_Pct), "
        "ACBL boards (Player_ID_N, Pct_NS), or canonical rows "
        "(player_a, player_b, date, pct)"
    )


def _prepare_sessions(
    sessions: pl.DataFrame,
    *,
    tau_days: Optional[float],
) -> pl.DataFrame:
    prepared = sessions.filter(
        pl.col("pct").is_not_null()
        & pl.col("pct").is_between(0.0, 100.0)
        & ~((pl.col("player_a") == "") & (pl.col("player_b") == ""))
        & (pl.col("player_a") != pl.col("player_b"))
    )
    if tau_days is not None:
        prepared = prepared.filter(pl.col("date").is_not_null())
    if prepared.is_empty():
        return prepared
    # One orientation per session so A-B and B-A are the same row.
    ordered = prepared.with_columns(
        pl.when(pl.col("player_a") <= pl.col("player_b"))
        .then(pl.col("player_a"))
        .otherwise(pl.col("player_b"))
        .alias("lo"),
        pl.when(pl.col("player_a") <= pl.col("player_b"))
        .then(pl.col("player_b"))
        .otherwise(pl.col("player_a"))
        .alias("hi"),
        pl.when(pl.col("player_a") <= pl.col("player_b"))
        .then(pl.col("elo_a"))
        .otherwise(pl.col("elo_b"))
        .alias("elo_lo"),
        pl.when(pl.col("player_a") <= pl.col("player_b"))
        .then(pl.col("elo_b"))
        .otherwise(pl.col("elo_a"))
        .alias("elo_hi"),
    ).select(
        pl.col("lo").alias("player_a"),
        pl.col("hi").alias("player_b"),
        "date",
        "pct",
        pl.col("elo_lo").alias("elo_a"),
        pl.col("elo_hi").alias("elo_b"),
    )
    return ordered


def _weights(dates: np.ndarray, tau_days: Optional[float]) -> np.ndarray:
    if tau_days is None or dates.size == 0:
        return np.ones(dates.size, dtype=np.float64)
    if tau_days <= 0:
        raise ValueError(f"tau_days must be positive or None, got {tau_days}")
    # datetime64[D]
    as_days = dates.astype("datetime64[D]")
    latest = as_days.max()
    age = (latest - as_days).astype(np.float64)
    return np.exp(-age / float(tau_days))


def _latest_elos(frame: pl.DataFrame) -> dict[str, float]:
    pieces = []
    for player_col, elo_col in (("player_a", "elo_a"), ("player_b", "elo_b")):
        pieces.append(
            frame.select(
                pl.col(player_col).alias("player_id"),
                pl.col("date"),
                pl.col(elo_col).alias("elo"),
            ).filter((pl.col("player_id") != "") & pl.col("elo").is_not_null())
        )
    long = pl.concat(pieces)
    if long.is_empty():
        return {}
    latest = (
        long.sort("date")
        .group_by("player_id")
        .agg(pl.col("elo").last())
    )
    return dict(zip(latest["player_id"].to_list(), latest["elo"].to_list(), strict=True))


def _calibrate_k(pair_elo: np.ndarray, logit: np.ndarray, weights: np.ndarray) -> Optional[float]:
    mask = np.isfinite(pair_elo) & np.isfinite(logit) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) < _CALIBRATE_MIN_ROWS:
        return None
    x = logit[mask]
    y = pair_elo[mask]
    w = weights[mask]
    sw = float(w.sum())
    if sw <= 0:
        return None
    mean_x = float(np.dot(w, x) / sw)
    mean_y = float(np.dot(w, y) / sw)
    var_x = float(np.dot(w, (x - mean_x) ** 2))
    if var_x <= 1e-12:
        return None
    cov = float(np.dot(w, (x - mean_x) * (y - mean_y)))
    k = cov / var_x
    if not math.isfinite(k) or k < 1.0:
        return None
    return k


def _fingerprint(sessions_df: pl.DataFrame, score: str, tau_days: Optional[float], ridge_lambda: float, r0: Optional[float], k_scale: Optional[float], min_sessions: int, min_partners: int) -> tuple:
    cols = [
        c
        for c in (
            "player1_id",
            "player2_id",
            "player_a",
            "player_b",
            "Player_ID_N",
            "Player_ID_S",
            "date",
            "Date",
            "pct",
            "National_Scratch_Pct",
            "National_Handicap_Pct",
            "Pct_NS",
            "session_id",
        )
        if c in sessions_df.columns
    ]
    if sessions_df.is_empty() or not cols:
        hashed: tuple = ()
    else:
        hashed = sessions_df.select(
            [pl.col(c).hash(seed=1).sum().alias(c) for c in cols]
        ).row(0)
    return (
        sessions_df.height,
        score,
        tau_days,
        ridge_lambda,
        r0,
        k_scale,
        min_sessions,
        min_partners,
        tuple(cols),
        hashed,
    )


def _cache_get(key: tuple) -> Optional[pl.DataFrame]:
    with _CACHE_LOCK:
        found = _CACHE.get(key)
        return None if found is None else found.clone()


def _cache_put(key: tuple, value: pl.DataFrame) -> None:
    with _CACHE_LOCK:
        if key not in _CACHE and len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = value.clone()


def compute_clone_ratings(
    sessions_df: pl.DataFrame,
    score: str = "Scratch",
    tau_days: Optional[float] = DEFAULT_TAU_DAYS,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    *,
    r0: Optional[float] = None,
    k_scale: Optional[float] = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    min_partners: int = DEFAULT_MIN_PARTNERS,
) -> pl.DataFrame:
    """Estimate clone ratings for every player in ``sessions_df``.

    ``tau_days=None`` is the unweighted career fit. Players below
    ``min_sessions`` or ``min_partners``, and players who cannot be separated
    from their only partner, keep ``Clone_N`` and ``Clone_Partners`` but have
    null ratings and ``Clone_Status`` of ``low sample`` or ``not identifiable``.
    """
    score = _require_score(score)
    if ridge_lambda < 0:
        raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
    if min_sessions < 1 or min_partners < 1:
        raise ValueError("min_sessions and min_partners must be >= 1")
    key = _fingerprint(
        sessions_df, score, tau_days, ridge_lambda, r0, k_scale, min_sessions, min_partners
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached

    started = datetime.now()
    canonical = _prepare_sessions(_canonical_sessions(sessions_df, score), tau_days=tau_days)
    if canonical.is_empty():
        result = empty_clone_ratings()
        _cache_put(key, result)
        return result

    dates = canonical["date"].to_numpy()
    weights = _weights(dates, tau_days)
    pct = canonical["pct"].to_numpy().astype(np.float64)
    logit = _logit_pct(pct)
    elo_a = canonical["elo_a"].to_numpy().astype(np.float64)
    elo_b = canonical["elo_b"].to_numpy().astype(np.float64)
    pair_elo = np.where(np.isfinite(elo_a) & np.isfinite(elo_b), (elo_a + elo_b) / 2.0, np.nan)
    latest = _latest_elos(canonical)
    if r0 is None:
        r0 = float(np.mean(list(latest.values()))) if latest else DEFAULT_R0
    if k_scale is None:
        k_scale = _calibrate_k(pair_elo, logit, weights) or K_SCALE_400
    if k_scale <= 0:
        raise ValueError(f"k_scale must be positive, got {k_scale}")

    y = k_scale * logit
    ids = sorted(
        {
            pid
            for pid in canonical["player_a"].to_list() + canonical["player_b"].to_list()
            if pid
        }
    )
    if not ids:
        result = empty_clone_ratings()
        _cache_put(key, result)
        return result
    index = {pid: i for i, pid in enumerate(ids)}
    lookup = pl.DataFrame({"player_id": ids, "ix": list(range(len(ids)))})
    mapped = (
        canonical.with_columns(
            pl.Series("y", y),
            pl.Series("sw", np.sqrt(weights)),
        )
        .join(lookup, left_on="player_a", right_on="player_id", how="left")
        .rename({"ix": "ia"})
        .join(lookup, left_on="player_b", right_on="player_id", how="left")
        .rename({"ix": "ib"})
    )
    n = mapped.height
    p = len(ids)
    ia = mapped["ia"].to_numpy()
    ib = mapped["ib"].to_numpy()
    sw = mapped["sw"].to_numpy()
    y = mapped["y"].to_numpy()
    row_ix = np.arange(n, dtype=np.int32)
    a_ok = np.isfinite(ia.astype(np.float64))
    b_ok = np.isfinite(ib.astype(np.float64))
    ia_i = np.where(a_ok, ia, -1).astype(np.int32)
    ib_i = np.where(b_ok, ib, -1).astype(np.int32)
    rows = np.concatenate([row_ix[a_ok], row_ix[b_ok]])
    cols = np.concatenate([ia_i[a_ok], ib_i[b_ok]])
    data = np.concatenate([sw[a_ok], sw[b_ok]])
    from scipy.sparse import coo_matrix, eye, vstack
    from scipy.sparse.linalg import lsqr

    design = coo_matrix((data, (rows, cols)), shape=(n, p)).tocsr()
    target = y * sw
    if ridge_lambda > 0:
        system = vstack([design, math.sqrt(ridge_lambda) * eye(p, format="csr")], format="csr")
        rhs = np.concatenate([target, np.zeros(p)])
    else:
        system = design
        rhs = target
    fit = lsqr(system, rhs, atol=1e-8, btol=1e-8, iter_lim=500)
    contribution = np.asarray(fit[0], dtype=np.float64)

    resid = y.copy()
    resid[a_ok] -= contribution[ia_i[a_ok]]
    resid[b_ok] -= contribution[ib_i[b_ok]]

    partners: dict[str, set[str]] = {pid: set() for pid in ids}
    edges = (
        mapped.filter((pl.col("player_a") != "") & (pl.col("player_b") != ""))
        .select("player_a", "player_b")
        .unique()
    )
    for a, b in zip(edges["player_a"].to_list(), edges["player_b"].to_list(), strict=True):
        partners[a].add(b)
        partners[b].add(a)
    unknown_partners = set(
        mapped.filter((pl.col("player_a") != "") & (pl.col("player_b") == ""))
        .get_column("player_a")
        .unique()
        .to_list()
    ) | set(
        mapped.filter((pl.col("player_b") != "") & (pl.col("player_a") == ""))
        .get_column("player_b")
        .unique()
        .to_list()
    )
    counted = pl.concat(
        [
            mapped.filter(pl.col("player_a") != "").select(pl.col("player_a").alias("player_id")),
            mapped.filter(pl.col("player_b") != "").select(pl.col("player_b").alias("player_id")),
        ]
    )
    counts = counted.group_by("player_id").len()
    session_count = dict(
        zip(counts["player_id"].to_list(), counts["len"].to_list(), strict=True)
    )
    inseparable: set[str] = set()
    for pid, mates in partners.items():
        if len(mates) != 1:
            continue
        other = next(iter(mates))
        if partners.get(other) == {pid}:
            inseparable.add(pid)

    seat_rows = pl.concat(
        [
            pl.DataFrame({"player_id": mapped["player_a"].to_list(), "resid": resid}).filter(
                pl.col("player_id") != ""
            ),
            pl.DataFrame({"player_id": mapped["player_b"].to_list(), "resid": resid}).filter(
                pl.col("player_id") != ""
            ),
        ]
    )
    grouped = seat_rows.group_by("player_id").agg(pl.col("resid").std(ddof=1).alias("sd"))
    resid_sd = {
        pid: sd
        for pid, sd in zip(grouped["player_id"].to_list(), grouped["sd"].to_list(), strict=True)
        if sd is not None and math.isfinite(sd)
    }

    clone_elo: list[Optional[int]] = []
    clone_pct: list[Optional[float]] = []
    clone_sd: list[Optional[float]] = []
    clone_n: list[int] = []
    clone_partners: list[int] = []
    partner_effect: list[Optional[int]] = []
    status: list[str] = []
    for pid in ids:
        n_sess = session_count[pid]
        n_partners = len(partners[pid]) + (1 if pid in unknown_partners else 0)
        c = float(contribution[index[pid]])
        rating = r0 + 2.0 * c
        z = (rating - r0) / k_scale
        prob = float(logistic(z))
        pct_value = 100.0 * prob
        slope = 100.0 * prob * (1.0 - prob) / k_scale
        sd_points = resid_sd.get(pid)
        sd_pct = None if sd_points is None else abs(slope) * sd_points
        current = latest.get(pid)
        effect = None if current is None else int(round(rating - current))
        if pid in inseparable:
            state = "not identifiable"
        elif n_sess < min_sessions or n_partners < min_partners:
            state = "low sample"
        else:
            state = "ok"
        publish = state == "ok"
        clone_n.append(n_sess)
        clone_partners.append(n_partners)
        status.append(state)
        if publish:
            clone_elo.append(int(np.clip(round(rating), 0, 3500)))
            clone_pct.append(round(pct_value, 1))
            clone_sd.append(None if sd_pct is None else round(sd_pct, 1))
            partner_effect.append(effect)
        else:
            clone_elo.append(None)
            clone_pct.append(None)
            clone_sd.append(None)
            partner_effect.append(None)

    result = pl.DataFrame(
        {
            "player_id": ids,
            "Clone_Elo": pl.Series("Clone_Elo", clone_elo, dtype=pl.Int64),
            "Clone_Pct": pl.Series("Clone_Pct", clone_pct, dtype=pl.Float64),
            "Clone_SD": pl.Series("Clone_SD", clone_sd, dtype=pl.Float64),
            "Clone_N": pl.Series("Clone_N", clone_n, dtype=pl.Int64),
            "Clone_Partners": pl.Series("Clone_Partners", clone_partners, dtype=pl.Int64),
            "Partner_Effect": pl.Series("Partner_Effect", partner_effect, dtype=pl.Int64),
            "Clone_Status": status,
        }
    )
    elapsed = (datetime.now() - started).total_seconds()
    if n >= 5000 or elapsed >= 30:
        print(
            f"[clone_elo] {score} fit {n} sessions, {p} players in {elapsed:.1f}s "
            f"(tau_days={tau_days}, lambda={ridge_lambda}, r0={r0:.1f}, k_scale={k_scale:.2f})",
            flush=True,
        )
    _cache_put(key, result)
    return result


def holdout_clone_mae(
    sessions_df: pl.DataFrame,
    *,
    score: str = "Scratch",
    holdout_fraction: float = 0.1,
    **kwargs: Any,
) -> dict[str, float]:
    """Fit on the earlier sessions and score percentage MAE on the latest slice.

    ``elo_mae`` predicts each held-out percentage from the average of the two
    stored Elos on the same logistic scale. It is omitted when those Elos are
    absent.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    tau_days = kwargs.get("tau_days", DEFAULT_TAU_DAYS)
    canonical = _prepare_sessions(
        _canonical_sessions(sessions_df, _require_score(score)),
        tau_days=tau_days,
    )
    if canonical.height < 20:
        raise ValueError("holdout needs at least 20 sessions")
    ordered = canonical.sort("date")
    cut = max(1, min(ordered.height - 1, int(round(ordered.height * (1.0 - holdout_fraction)))))
    train = ordered.head(cut)
    test = ordered.tail(ordered.height - cut)
    loose = dict(kwargs)
    loose["min_sessions"] = 1
    loose["min_partners"] = 1
    ratings = compute_clone_ratings(train, score=score, **loose)
    r0 = kwargs.get("r0")
    k_scale = kwargs.get("k_scale")
    prepared = train
    weights = _weights(prepared["date"].to_numpy(), tau_days)
    logit = _logit_pct(prepared["pct"].to_numpy().astype(np.float64))
    elo_a = prepared["elo_a"].to_numpy().astype(np.float64)
    elo_b = prepared["elo_b"].to_numpy().astype(np.float64)
    pair_elo = np.where(np.isfinite(elo_a) & np.isfinite(elo_b), (elo_a + elo_b) / 2.0, np.nan)
    latest = _latest_elos(prepared)
    if r0 is None:
        r0 = float(np.mean(list(latest.values()))) if latest else DEFAULT_R0
    if k_scale is None:
        k_scale = _calibrate_k(pair_elo, logit, weights) or K_SCALE_400
    by_id = {row["player_id"]: row for row in ratings.to_dicts()}

    def _c(pid: str) -> float:
        row = by_id.get(pid)
        if not pid or row is None or row["Clone_Elo"] is None:
            return 0.0
        return (float(row["Clone_Elo"]) - float(r0)) / 2.0

    abs_err = []
    elo_err = []
    for row in test.to_dicts():
        pred = 100.0 * float(logistic((_c(row["player_a"]) + _c(row["player_b"])) / k_scale))
        abs_err.append(abs(pred - float(row["pct"])))
        if row["elo_a"] is not None and row["elo_b"] is not None:
            mid = (float(row["elo_a"]) + float(row["elo_b"])) / 2.0
            elo_pred = 100.0 * float(logistic((mid - r0) / k_scale))
            elo_err.append(abs(elo_pred - float(row["pct"])))
    out = {"clone_mae": float(np.mean(abs_err)), "n_holdout": float(len(abs_err))}
    if elo_err:
        out["elo_mae"] = float(np.mean(elo_err))
    return out


def acbl_sessions_from_connection(con: Any, elo_columns: dict[str, Optional[str]]) -> pl.DataFrame:
    """Aggregate the DuckDB view ``self`` to one ACBL pair-session per direction."""
    names = {row[0] for row in con.execute("DESCRIBE self").fetchall()}
    if "Pct_NS" not in names or "session_id" not in names or "Date" not in names:
        raise ValueError("ACBL self view is missing Pct_NS, session_id, or Date")
    pattern = elo_columns.get("player_pattern")

    def _elo(pos: str) -> str:
        if not pattern:
            return "NULL"
        column = pattern.format(pos=pos)
        if not column.replace("_", "").isalnum():
            raise ValueError(f"Unexpected Elo column {column!r}")
        if column not in names:
            return "NULL"
        return f"AVG({column})"

    sql = f"""
        SELECT player_a, player_b, date, pct, elo_a, elo_b FROM (
            SELECT
                CAST(Player_ID_N AS VARCHAR) AS player_a,
                CAST(Player_ID_S AS VARCHAR) AS player_b,
                CAST(Date AS DATE) AS date,
                AVG(Pct_NS) * 100 AS pct,
                {_elo("N")} AS elo_a,
                {_elo("S")} AS elo_b
            FROM self
            WHERE Pct_NS IS NOT NULL AND Pct_NS >= 0 AND Pct_NS <= 1
              AND Player_ID_N IS NOT NULL AND Player_ID_S IS NOT NULL
            GROUP BY 1, 2, 3, session_id
            UNION ALL
            SELECT
                CAST(Player_ID_E AS VARCHAR) AS player_a,
                CAST(Player_ID_W AS VARCHAR) AS player_b,
                CAST(Date AS DATE) AS date,
                AVG(1.0 - Pct_NS) * 100 AS pct,
                {_elo("E")} AS elo_a,
                {_elo("W")} AS elo_b
            FROM self
            WHERE Pct_NS IS NOT NULL AND Pct_NS >= 0 AND Pct_NS <= 1
              AND Player_ID_E IS NOT NULL AND Player_ID_W IS NOT NULL
            GROUP BY 1, 2, 3, session_id
        )
    """
    return con.execute(sql).pl()
