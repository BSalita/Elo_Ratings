"""Clone Elo: the rating a player would have partnered with a copy of themselves.

A pair rating is ``R0 + c_A + c_B``. ``Clone_Elo(A) = R0 + 2·c_A``. Contributions
come from a ridge regression of session log-odds over every pair session, so a
partner's strength is measured by how that partner does with everyone else.

``k_scale`` converts log-odds to rating points. When the session rows carry
current Elo and ``k_scale`` is not given, ``Clone_Elo`` for ``ok`` players is
matched to the mean and SD of those players' current Elo, so ``Clone_Elo`` and
``Partner_Effect`` share the stored Elo scale. Otherwise the classic
``400/ln(10)`` is used.
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
# Best held-out MAE on FFBridge 2023-2026 scratch and handicap (tried 1, 3, 10, 30, 100).
DEFAULT_RIDGE_LAMBDA = 3.0
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
            | lowered.is_in(["none", "nan", "null", "<na>", "0"])
        )
        .then(pl.lit(""))
        .otherwise(text.str.replace(r"\.0$", ""))
        .alias(column)
    )


def _date_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    """Calendar date from Date, Datetime, or ISO text such as ``2026-09-28T00:00:00+02:00``."""
    dtype = frame.schema[column]
    source = pl.col(column)
    if dtype == pl.Date:
        expr = source
    elif isinstance(dtype, pl.Datetime):
        expr = source.dt.date()
    elif dtype == pl.Utf8:
        expr = source.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=True)
    else:
        raise ValueError(f"Clone sessions need a date column, got {column} as {dtype}")
    return expr.alias("date")


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
        _date_expr(sessions_df, "date"),
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
                _date_expr(boards, "Date"),
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
            _date_expr(sessions_df, date_col),
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
    # Exactly 0 or 100 is an unscored placeholder (FFBridge handicap shells), not a result.
    prepared = sessions.filter(
        pl.col("pct").is_not_null()
        & (pl.col("pct") > 0.0)
        & (pl.col("pct") < 100.0)
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


def _match_scale(two_c: np.ndarray, current: np.ndarray) -> Optional[tuple[float, float]]:
    """``(k_scale, r0)`` so ``r0 + k_scale * two_c`` has the mean and SD of ``current``."""
    mask = np.isfinite(two_c) & np.isfinite(current)
    if int(mask.sum()) < _CALIBRATE_MIN_ROWS:
        return None
    x = two_c[mask]
    y = current[mask]
    sd_x = float(x.std())
    sd_y = float(y.std())
    if sd_x <= 1e-9 or sd_y <= 1e-9:
        return None
    k = sd_y / sd_x
    return k, float(y.mean()) - k * float(x.mean())


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


class _Fit:
    """Ridge contributions in log-odds units for one prepared session frame."""

    def __init__(self, canonical: pl.DataFrame, tau_days: Optional[float], ridge_lambda: float) -> None:
        from scipy.sparse import coo_matrix, eye, vstack
        from scipy.sparse.linalg import lsqr

        self.ids = sorted(
            {
                pid
                for pid in canonical["player_a"].to_list() + canonical["player_b"].to_list()
                if pid
            }
        )
        self.index = {pid: i for i, pid in enumerate(self.ids)}
        weights = _weights(canonical["date"].to_numpy(), tau_days)
        lookup = pl.DataFrame(
            {"player_id": self.ids, "ix": list(range(len(self.ids)))},
            schema={"player_id": pl.Utf8, "ix": pl.Int64},
        )
        mapped = (
            canonical.with_columns(
                pl.Series("y", _logit_pct(canonical["pct"].to_numpy().astype(np.float64))),
                pl.Series("sw", np.sqrt(weights)),
            )
            .join(lookup, left_on="player_a", right_on="player_id", how="left")
            .rename({"ix": "ia"})
            .join(lookup, left_on="player_b", right_on="player_id", how="left")
            .rename({"ix": "ib"})
        )
        self.mapped = mapped
        n = mapped.height
        p = len(self.ids)
        ia = mapped["ia"].fill_null(-1).to_numpy().astype(np.int64)
        ib = mapped["ib"].fill_null(-1).to_numpy().astype(np.int64)
        sw = mapped["sw"].to_numpy()
        y = mapped["y"].to_numpy()
        a_ok = ia >= 0
        b_ok = ib >= 0
        row_ix = np.arange(n, dtype=np.int64)
        self.contribution = np.zeros(p, dtype=np.float64)
        self.resid = y.copy()
        if p == 0:
            return
        design = coo_matrix(
            (
                np.concatenate([sw[a_ok], sw[b_ok]]),
                (
                    np.concatenate([row_ix[a_ok], row_ix[b_ok]]),
                    np.concatenate([ia[a_ok], ib[b_ok]]),
                ),
            ),
            shape=(n, p),
        ).tocsr()
        target = y * sw
        if ridge_lambda > 0:
            system = vstack([design, math.sqrt(ridge_lambda) * eye(p, format="csr")], format="csr")
            rhs = np.concatenate([target, np.zeros(p)])
        else:
            system = design
            rhs = target
        solved = lsqr(system, rhs, atol=1e-10, btol=1e-10, iter_lim=2000)
        self.contribution = np.asarray(solved[0], dtype=np.float64)
        self.resid[a_ok] -= self.contribution[ia[a_ok]]
        self.resid[b_ok] -= self.contribution[ib[b_ok]]

    def c(self, pid: str) -> float:
        i = self.index.get(pid)
        return 0.0 if i is None else float(self.contribution[i])


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
    if k_scale is not None and k_scale <= 0:
        raise ValueError(f"k_scale must be positive, got {k_scale}")
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

    fit = _Fit(canonical, tau_days, ridge_lambda)
    ids = fit.ids
    if not ids:
        result = empty_clone_ratings()
        _cache_put(key, result)
        return result
    mapped = fit.mapped
    resid = fit.resid

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
        if len(mates) != 1 or pid in unknown_partners:
            continue
        other = next(iter(mates))
        if partners.get(other) == {pid} and other not in unknown_partners:
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

    status: list[str] = []
    clone_n: list[int] = []
    clone_partners: list[int] = []
    for pid in ids:
        n_sess = session_count[pid]
        n_partners = len(partners[pid]) + (1 if pid in unknown_partners else 0)
        if pid in inseparable:
            state = "not identifiable"
        elif n_sess < min_sessions or n_partners < min_partners:
            state = "low sample"
        else:
            state = "ok"
        status.append(state)
        clone_n.append(n_sess)
        clone_partners.append(n_partners)

    two_c = 2.0 * fit.contribution
    latest = _latest_elos(canonical)
    current = np.array([latest.get(pid, np.nan) for pid in ids], dtype=np.float64)
    publish = np.array([state == "ok" for state in status])
    scale = "given"
    if k_scale is None:
        matched = _match_scale(two_c[publish], current[publish])
        if matched is None:
            k_scale = K_SCALE_400
            scale = "classic"
        else:
            k_scale, matched_r0 = matched
            scale = "matched"
            if r0 is None:
                r0 = matched_r0
    if r0 is None:
        finite = current[np.isfinite(current)]
        r0 = float(finite.mean()) if finite.size else DEFAULT_R0

    rating = r0 + k_scale * two_c
    prob = logistic(two_c)
    slope = 100.0 * prob * (1.0 - prob)
    clone_elo: list[Optional[int]] = []
    clone_pct: list[Optional[float]] = []
    clone_sd: list[Optional[float]] = []
    partner_effect: list[Optional[int]] = []
    for i, pid in enumerate(ids):
        if not publish[i]:
            clone_elo.append(None)
            clone_pct.append(None)
            clone_sd.append(None)
            partner_effect.append(None)
            continue
        clone_elo.append(int(np.clip(round(rating[i]), 0, 3500)))
        clone_pct.append(round(100.0 * float(prob[i]), 1))
        sd_logit = resid_sd.get(pid)
        clone_sd.append(None if sd_logit is None else round(float(slope[i]) * sd_logit, 1))
        partner_effect.append(
            None if not math.isfinite(current[i]) else int(round(rating[i] - current[i]))
        )

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
    n = mapped.height
    if n >= 5000 or elapsed >= 30:
        print(
            f"[clone_elo] {score} fit {n} sessions, {len(ids)} players, "
            f"{int(publish.sum())} ok in {elapsed:.1f}s (tau_days={tau_days}, "
            f"lambda={ridge_lambda}, r0={r0:.1f}, k_scale={k_scale:.2f} {scale})",
            flush=True,
        )
    _cache_put(key, result)
    return result


def holdout_clone_mae(
    sessions_df: pl.DataFrame,
    *,
    score: str = "Scratch",
    holdout_fraction: float = 0.1,
    tau_days: Optional[float] = DEFAULT_TAU_DAYS,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
) -> dict[str, float]:
    """Fit on the earlier sessions and score percentage MAE on the latest slice.

    The clone prediction is ``logistic(c_A + c_B)`` with unseen players at 0.
    ``elo_mae`` uses each player's last Elo in the training slice, averaged per
    pair and mapped to percentage by a log-odds line fitted on the training
    sessions. Held-out rows' own Elos are not used because they already include
    that session. It is omitted when Elos are absent.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
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
    fit = _Fit(train, tau_days, ridge_lambda)

    lookup = pl.DataFrame(
        {"player_id": fit.ids, "c": fit.contribution},
        schema={"player_id": pl.Utf8, "c": pl.Float64},
    )
    scored = (
        test.join(lookup.rename({"player_id": "player_a", "c": "c_a"}), on="player_a", how="left")
        .join(lookup.rename({"player_id": "player_b", "c": "c_b"}), on="player_b", how="left")
        .with_columns(pl.col("c_a").fill_null(0.0), pl.col("c_b").fill_null(0.0))
    )
    actual = scored["pct"].to_numpy().astype(np.float64)
    clone_pred = 100.0 * logistic(scored["c_a"].to_numpy() + scored["c_b"].to_numpy())
    out = {
        "clone_mae": float(np.mean(np.abs(clone_pred - actual))),
        "n_holdout": float(scored.height),
    }

    last_elo = _latest_elos(train)

    def _pair_elo(frame: pl.DataFrame) -> np.ndarray:
        a = np.array([last_elo.get(pid, np.nan) for pid in frame["player_a"].to_list()], dtype=np.float64)
        b = np.array([last_elo.get(pid, np.nan) for pid in frame["player_b"].to_list()], dtype=np.float64)
        return np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2.0, np.nan)

    train_elo = _pair_elo(train)
    train_logit = _logit_pct(train["pct"].to_numpy().astype(np.float64))
    train_w = _weights(train["date"].to_numpy(), tau_days)
    mask = np.isfinite(train_elo)
    test_elo = _pair_elo(scored)
    test_mask = np.isfinite(test_elo)
    if int(mask.sum()) >= _CALIBRATE_MIN_ROWS and test_mask.any():
        x = train_elo[mask]
        yv = train_logit[mask]
        w = train_w[mask]
        mean_x = float(np.average(x, weights=w))
        mean_y = float(np.average(yv, weights=w))
        var_x = float(np.average((x - mean_x) ** 2, weights=w))
        beta = 0.0 if var_x <= 1e-12 else float(np.average((x - mean_x) * (yv - mean_y), weights=w)) / var_x
        elo_pred = 100.0 * logistic(mean_y + beta * (test_elo[test_mask] - mean_x))
        out["elo_mae"] = float(np.mean(np.abs(elo_pred - actual[test_mask])))
        out["clone_mae_same_rows"] = float(np.mean(np.abs(clone_pred[test_mask] - actual[test_mask])))
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
