"""Headless FFBridge Elo report service.

Single source of truth for the FFBridge leaderboard logic, used by the
Streamlit app and first-party FFBridge REST API. MCP calls that REST API.
Reads the persisted Elo parquet set (written by build_ffbridge_elo_parquets.py
/ compute_and_persist_elo_dataset) and runs the leaderboard SQL. No Streamlit
imports here.
"""

import json
import os
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import polars as pl

from elo_filter_common import (
    filter_ffbridge_leaderboard,
    filter_fuzzy_text,
    filter_normalized_substring,
    fuzzy_text_score,
)

# Directory for the persisted (precomputed) Elo dataset parquets. In production
# set FFBRIDGE_CACHE_DIR to a persistent mount (e.g. /data/ffbridge) so the raw
# tournament cache and elo_cache parquet survive redeploys. Locally
# (FFBRIDGE_CACHE_DIR unset) we fall back to the app dir.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_CACHE_DIR_ENV = os.environ.get("FFBRIDGE_CACHE_DIR", "").strip()
if _CACHE_DIR_ENV:
    ELO_CACHE_DIR = pathlib.Path(_CACHE_DIR_ENV).resolve() / "elo_cache"
else:
    ELO_CACHE_DIR = _SCRIPT_DIR / "data" / "ffbridge" / "elo_cache"

# ACBL-style rolling windows, plus FFBridge season years (ratings reset July 1).
DATE_RANGE_OPTIONS = (
    "All time",
    "Current FFBridge year",
    "Previous FFBridge year",
    "Last 3 months",
    "Last 6 months",
    "Last 1 year",
    "Last 2 years",
    "Last 3 years",
    "Last 4 years",
    "Last 5 years",
)

# Leaderboard defaults, kept in sync with the Streamlit URL-param defaults.
DEFAULT_TOP_N = 250
DEFAULT_MIN_GAMES = 10
DEFAULT_PRIOR_SESSIONS = 50

API_BACKEND_KEYS = {
    "FFBridge Classic API": "FFBridge_Classic_API",
    "FFBridge Lancelot API": "FFBridge_Lancelot_API",
}

SERIES_NAMES = {
    3: "Rondes de France",
    4: "Trophes du Voyage",
    5: "Roy Rene",
    140: "Armour du Bridge",
    384: "Simultanet",
    386: "Simultane Octopus",
    604: "Atout Simultane",
    868: "Festival des Simultanes",
}


def resolve_series_id(series: Optional[int | str]) -> Optional[int]:
    """Resolve an exact ID or fuzzy tournament-series name."""
    if series in (None, "", "all"):
        return None
    try:
        series_id = int(series)
    except (TypeError, ValueError):
        ranked = sorted(
            (
                (fuzzy_text_score(name, series), series_id, name)
                for series_id, name in SERIES_NAMES.items()
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.72:
            raise ValueError(
                f"Unknown tournament series {series!r}; valid: {SERIES_NAMES}"
            )
        return ranked[0][1]
    if series_id not in SERIES_NAMES:
        raise ValueError(f"Unknown series_id {series_id}; valid: {list(SERIES_NAMES)}")
    return series_id


def default_api_key() -> str:
    """Persisted-parquet key for the default backend (mirrors the UI default)."""
    prefer_classic = os.getenv("FFBRIDGE_PREFER_CLASSIC_API", "").strip().lower() in (
        "1", "true", "yes",
    )
    if prefer_classic and os.getenv("FFBRIDGE_BEARER_TOKEN", "").strip():
        return "FFBridge_Classic_API"
    return "FFBridge_Lancelot_API"


def api_key_for_backend(api_backend: Optional[str]) -> str:
    """Convert a Streamlit backend label (or persisted key) to a cache key."""
    if not api_backend:
        return default_api_key()
    value = str(api_backend).strip()
    if value in API_BACKEND_KEYS:
        return API_BACKEND_KEYS[value]
    if value in API_BACKEND_KEYS.values():
        return value
    raise ValueError(
        f"Unknown api_backend {api_backend!r}; valid: {list(API_BACKEND_KEYS)}"
    )


# -------------------------------
# Persisted parquet-set resolution
# -------------------------------
def elo_cache_key(api_key: str, fetch_iv: bool, n_tournaments: int = 0) -> str:
    """Stable parquet identity per backend and IV mode.

    Tournament-list length (including scheduled future sessions) is intentionally
    excluded: embedding it forced a full rebuild whenever the API added future
    sessions even when no new past events existed.
    """
    del n_tournaments
    return f"elo_full_v4_{api_key}_iv_{int(fetch_iv)}"


def legacy_elo_cache_keys(api_key: str, fetch_iv: bool) -> List[str]:
    """Parquet keys from before the stable-key change (middle segment = list length)."""
    prefix = f"elo_full_v4_{api_key}_"
    suffix = f"_iv_{int(fetch_iv)}"
    keys: List[str] = []
    for meta_path in ELO_CACHE_DIR.glob(f"{prefix}*{suffix}.meta.json"):
        key = meta_path.name[: -len(".meta.json")]
        middle = key[len(prefix): -len(suffix)]
        if middle.isdigit():
            keys.append(key)
    return keys


def elo_cache_paths(key: str) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    return (
        ELO_CACHE_DIR / f"{key}.results.parquet",
        ELO_CACHE_DIR / f"{key}.players.parquet",
        ELO_CACHE_DIR / f"{key}.meta.json",
    )


def resolve_elo_cache_key(api_key: str, fetch_iv: bool) -> Optional[str]:
    """Best on-disk parquet set: stable key first, else newest legacy count-keyed set."""
    stable = elo_cache_key(api_key, fetch_iv)
    results_path, players_path, meta_path = elo_cache_paths(stable)
    if results_path.exists() and players_path.exists() and meta_path.exists():
        return stable

    newest_key: Optional[str] = None
    newest_dt: Optional[datetime] = None
    for key in legacy_elo_cache_keys(api_key, fetch_iv):
        results_path, players_path, meta_path = elo_cache_paths(key)
        if not (results_path.exists() and players_path.exists() and meta_path.exists()):
            continue
        try:
            built_at = json.loads(meta_path.read_text(encoding="utf-8")).get("built_at")
            if not built_at:
                continue
            dt = datetime.fromisoformat(built_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if newest_dt is None or dt > newest_dt:
                newest_dt = dt
                newest_key = key
        except Exception:
            continue
    return newest_key


# Per (api_key, fetch_iv): (cache_key, results_mtime, results_df, meta). Reloads
# when a rebuild replaces the parquet (mtime change) or resolves to a new key.
_RESULTS_CACHE: Dict[Tuple[str, bool], Tuple[str, float, pl.DataFrame, Dict[str, Any]]] = {}


def load_results(
    api_key: Optional[str] = None,
    fetch_iv: bool = True,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Persisted result rows + meta for a backend. Fails fast when absent."""
    key_api = api_key or default_api_key()
    key = resolve_elo_cache_key(key_api, fetch_iv)
    if key is None:
        raise FileNotFoundError(
            f"No persisted FFBridge Elo parquet set for api_key={key_api!r} "
            f"fetch_iv={fetch_iv} under {ELO_CACHE_DIR}. Run "
            "build_ffbridge_elo_parquets.py (or let ffbridge-elo build it) first."
        )
    results_path, _players_path, meta_path = elo_cache_paths(key)
    mtime = results_path.stat().st_mtime
    cached = _RESULTS_CACHE.get((key_api, fetch_iv))
    if cached is not None and cached[0] == key and cached[1] == mtime:
        return cached[2], cached[3]
    results_df = pl.read_parquet(results_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _RESULTS_CACHE[(key_api, fetch_iv)] = (key, mtime, results_df, meta)
    print(
        f"[ffbridge_report_service] loaded '{key}' ({results_df.height} result rows)",
        flush=True,
    )
    return results_df, meta


def dataset_info(api_key: Optional[str] = None, fetch_iv: bool = True) -> Dict[str, Any]:
    """Summary of the persisted dataset (no result rows)."""
    results_df, meta = load_results(api_key, fetch_iv)
    clubs: List[str] = []
    if "club_name" in results_df.columns:
        clubs = sorted(
            {
                str(c).strip()
                for c in results_df.get_column("club_name").unique().to_list()
                if c is not None and str(c).strip()
            }
        )
    date_min = date_max = None
    if "date" in results_df.columns and not results_df.is_empty():
        day = results_df.get_column("date").cast(pl.Utf8).str.slice(0, 10)
        date_min, date_max = day.min(), day.max()
    return {
        "api_key": api_key or default_api_key(),
        "built_at": meta.get("built_at"),
        "result_rows": results_df.height,
        "date_min": date_min,
        "date_max": date_max,
        "clubs": clubs,
        "processing_stats": meta.get("processing_stats", {}),
        "score_provenance": score_provenance_counts(results_df),
        "date_range_options": list(DATE_RANGE_OPTIONS),
        "api_backends": list(API_BACKEND_KEYS),
        "tournament_series": [
            {"series_id": series_id, "name": name}
            for series_id, name in SERIES_NAMES.items()
        ],
    }


# -------------------------------
# Date ranges
# -------------------------------
def ffbridge_season_july1(today: datetime) -> datetime:
    """Start of the FFBridge year that contains ``today`` (July 1)."""
    year = today.year if today.month >= 7 else today.year - 1
    return today.replace(year=year, month=7, day=1, hour=0, minute=0, second=0, microsecond=0)


def date_range_bounds(date_range_choice: str) -> Tuple[Optional[str], Optional[str]]:
    """Inclusive YYYY-MM-DD (from, to) for Date range; None means unbounded on that side."""
    now = datetime.now()
    today_s = now.strftime("%Y-%m-%d")
    if date_range_choice == "All time":
        return None, None
    if date_range_choice == "Current FFBridge year":
        # July 1 of the active season through today.
        return ffbridge_season_july1(now).strftime("%Y-%m-%d"), today_s
    if date_range_choice == "Previous FFBridge year":
        # Prior season: July 1 through May 31 (FFBridge season window).
        current_july1 = ffbridge_season_july1(now)
        prev_july1 = current_july1.replace(year=current_july1.year - 1)
        prev_may31 = current_july1.replace(month=5, day=31)
        return prev_july1.strftime("%Y-%m-%d"), prev_may31.strftime("%Y-%m-%d")
    days = {
        "Last 3 months": 90,
        "Last 6 months": 182,
        "Last 1 year": 365,
        "Last 2 years": 365 * 2,
        "Last 3 years": 365 * 3,
        "Last 4 years": 365 * 4,
        "Last 5 years": 365 * 5,
    }.get(date_range_choice)
    if days is None:
        return None, None
    return (now - timedelta(days=days)).strftime("%Y-%m-%d"), None


# -------------------------------
# Filtering
# -------------------------------
def filter_valid_percentages(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with invalid percentage values (<0 or >100)."""
    if df.is_empty():
        return df

    pct_cols = [c for c in ("percentage", "scratch_percentage", "handicap_percentage", "club_percentage") if c in df.columns]
    if not pct_cols:
        return df

    valid_expr = pl.lit(True)
    for col_name in pct_cols:
        col = pl.col(col_name).cast(pl.Float64, strict=False)
        valid_expr = valid_expr & (col.is_null() | ((col >= 0.0) & (col <= 100.0)))

    return df.filter(valid_expr)


def filter_score_available(df: pl.DataFrame, use_handicap: bool) -> pl.DataFrame:
    """Exclude only categories explicitly unresolved during publication."""
    status_column = (
        "handicap_score_status" if use_handicap else "scratch_score_status"
    )
    if df.is_empty() or status_column not in df.columns:
        return df
    return df.filter(
        pl.col(status_column).is_null()
        | (pl.col(status_column).cast(pl.Utf8) != "unresolved")
    )


def score_provenance_counts(df: pl.DataFrame) -> Dict[str, int]:
    """Compact score-source diagnostics for API/UI metadata."""
    if df.is_empty():
        return {
            "official_rows": 0,
            "provisional_rows": 0,
            "unresolved_rows": 0,
            "provisional_scratch_rows": 0,
            "provisional_handicap_rows": 0,
            "unresolved_scratch_rows": 0,
            "unresolved_handicap_rows": 0,
        }

    def _count(column: str, value: str) -> int:
        if column not in df.columns:
            return 0
        return df.filter(pl.col(column).cast(pl.Utf8) == value).height

    return {
        "official_rows": _count("score_status", "official"),
        "provisional_rows": _count("score_status", "provisional"),
        "unresolved_rows": _count("score_status", "unresolved"),
        "provisional_scratch_rows": _count(
            "scratch_score_status", "provisional"
        ),
        "provisional_handicap_rows": _count(
            "handicap_score_status", "provisional"
        ),
        "unresolved_scratch_rows": _count(
            "scratch_score_status", "unresolved"
        ),
        "unresolved_handicap_rows": _count(
            "handicap_score_status", "unresolved"
        ),
    }


def filter_results(
    results_df: pl.DataFrame,
    *,
    series_id: Optional[int | str] = None,
    tournament: Optional[str] = None,
    tournament_contains: Optional[str] = None,
    club: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pl.DataFrame:
    """Pre-ranking filters for tournament, series, club, and date."""
    if tournament and tournament_contains:
        raise ValueError(
            "Pass either tournament or tournament_contains, not both"
        )
    df = results_df
    normalized_series_id = resolve_series_id(series_id)
    if normalized_series_id is not None and "series_id" in df.columns:
        df = df.filter(
            pl.col("series_id").cast(pl.Int64, strict=False) == normalized_series_id
        )
    if tournament or tournament_contains:
        if "tournament_name" not in df.columns:
            raise ValueError("Persisted results do not contain tournament_name")
        tournament_names = pl.col("tournament_name").cast(pl.Utf8)
        if tournament:
            df = df.filter(tournament_names == tournament.strip())
        else:
            df = filter_normalized_substring(
                df,
                column="tournament_name",
                query=tournament_contains,
            )
    if club and not df.is_empty() and "club_name" in df.columns:
        df = filter_fuzzy_text(df, column="club_name", query=club)
    if (date_from or date_to) and not df.is_empty() and "date" in df.columns:
        session_day = pl.col("date").cast(pl.Utf8).str.slice(0, 10)
        if date_from:
            df = df.filter(session_day >= date_from)
        if date_to:
            df = df.filter(session_day <= date_to)
    return df


def list_tournaments(
    *,
    club: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    contains: Optional[str] = None,
    limit: int = 500,
    api_key: Optional[str] = None,
    fetch_iv: bool = True,
) -> Dict[str, Any]:
    """Discover canonical tournament names after optional club/date filters."""
    results_df, meta = load_results(api_key, fetch_iv)
    results_df = filter_valid_percentages(results_df)
    results_df = filter_results(
        results_df,
        club=club,
        date_from=date_from,
        date_to=date_to,
    )
    if "tournament_name" not in results_df.columns:
        raise ValueError("Persisted results do not contain tournament_name")
    if contains:
        results_df = filter_normalized_substring(
            results_df,
            column="tournament_name",
            query=contains,
        )
    grouping = ["tournament_name"]
    if "series_id" in results_df.columns:
        grouping.append("series_id")
    rows = (
        results_df.group_by(grouping)
        .agg(
            pl.len().alias("result_rows"),
            pl.col("date").cast(pl.Utf8).str.slice(0, 10).min().alias("date_min"),
            pl.col("date").cast(pl.Utf8).str.slice(0, 10).max().alias("date_max"),
            pl.col("club_name").n_unique().alias("club_count"),
        )
        .sort(["date_max", "tournament_name"], descending=[True, False])
        .head(limit)
        .to_dicts()
    )
    return {
        "tournaments": rows,
        "count": len(rows),
        "club": club,
        "date_from": date_from,
        "date_to": date_to,
        "contains": contains,
        "dataset_built_at": meta.get("built_at"),
    }


# -------------------------------
# Aggregation + leaderboard SQL
# -------------------------------
def aggregate_players_from_results(results_df: pl.DataFrame, use_handicap: bool) -> pl.DataFrame:
    """Aggregate per-player stats from filtered result rows (duckdb)."""
    if results_df.is_empty():
        return pl.DataFrame()
    elo_col_p1 = "player1_handicap_elo_after" if use_handicap else "player1_scratch_elo_after"
    elo_col_p2 = "player2_handicap_elo_after" if use_handicap else "player2_scratch_elo_after"
    pct_expr = "COALESCE(handicap_percentage, scratch_percentage)" if use_handicap else "scratch_percentage"
    return duckdb.sql(f"""
        WITH player_results AS (
            SELECT
                player1_id AS player_id,
                player1_name AS player_name,
                player1_scratch_elo_after AS scratch_elo,
                player1_handicap_elo_after AS handicap_elo,
                {elo_col_p1} AS elo_rating,
                scratch_percentage,
                handicap_percentage,
                iv_bonus,
                score_status,
                date
            FROM results_df
            UNION ALL
            SELECT
                player2_id AS player_id,
                player2_name AS player_name,
                player2_scratch_elo_after AS scratch_elo,
                player2_handicap_elo_after AS handicap_elo,
                {elo_col_p2} AS elo_rating,
                scratch_percentage,
                handicap_percentage,
                iv_bonus,
                score_status,
                date
            FROM results_df
        )
        SELECT
            player_id,
            ARG_MAX(player_name, date) AS player_name,
            ROUND(ARG_MAX(scratch_elo, date), 1) AS scratch_elo,
            ROUND(ARG_MAX(COALESCE(handicap_elo, scratch_elo), date), 1) AS handicap_elo,
            ROUND(ARG_MAX(elo_rating, date), 1) AS elo_rating,
            COUNT(*) AS games_played,
            SUM(CASE WHEN score_status = 'provisional' THEN 1 ELSE 0 END)
                AS provisional_games,
            ROUND(AVG(scratch_percentage), 2) AS avg_scratch_pct,
            ROUND(AVG(COALESCE(handicap_percentage, scratch_percentage)), 2) AS avg_handicap_pct,
            ROUND(AVG(iv_bonus), 1) AS avg_iv_bonus,
            ROUND(AVG({pct_expr}), 2) AS avg_percentage,
            ROUND(STDDEV_SAMP({pct_expr}), 2) AS stdev_percentage
        FROM player_results
        GROUP BY player_id
    """).pl()


def show_top_players(
    players_df: pl.DataFrame,
    top_n: int,
    min_games: int = 5,
    use_handicap: bool = False,
    prior_sessions: int = 0,
) -> Tuple[pl.DataFrame, str, Optional[float]]:
    """Get top players sorted by Elo rating using SQL.

    When ``prior_sessions > 0`` and a prior anchor (median Elo of the
    qualifying subset) is available, the headline Elo is the Bayesian-shrunk
    "Published" Elo:

        Published = (games * Raw + prior_sessions * prior_anchor)
                    / (games + prior_sessions)

    Both the Published and Raw values are returned in the table; the
    leaderboard is ordered by Published. Returns ``(df, sql, prior_anchor)``.
    """
    if players_df.is_empty():
        return players_df, "", None

    elo_col_name = "HC_Player_Elo" if use_handicap else "Player_Elo"
    title_col_name = "Scratch_Title" if use_handicap else "Title"

    anchor_query = f"""
        SELECT
            MEDIAN(elo_rating) AS anchor,
            MEDIAN(scratch_elo) AS scratch_anchor
        FROM players_df
        WHERE games_played >= {min_games}
    """
    anchor_df = duckdb.sql(anchor_query).pl()
    if anchor_df.is_empty():
        prior_anchor: Optional[float] = None
        scratch_anchor: Optional[float] = None
    else:
        val = anchor_df.item(0, 0)
        prior_anchor = float(val) if val is not None else None
        sval = anchor_df.item(0, 1)
        scratch_anchor = float(sval) if sval is not None else None

    if prior_sessions > 0 and prior_anchor is not None:
        ps_lit = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
        anchor_lit = f"CAST({float(prior_anchor)!r} AS DOUBLE)"
        published_expr = (
            f"CAST(ROUND(LEAST(GREATEST("
            f"(CAST(games_played AS DOUBLE) * CAST(elo_rating AS DOUBLE) "
            f"+ {ps_lit} * {anchor_lit}) "
            f"/ NULLIF(CAST(games_played AS DOUBLE) + {ps_lit}, 0)"
            f", 0), 3500), 0) AS INTEGER)"
        )
    else:
        published_expr = "CAST(ROUND(LEAST(GREATEST(elo_rating, 0), 3500), 0) AS INTEGER)"

    # Titles derive from the *published* (Bayesian-shrunk) SCRATCH Elo so they
    # always agree with the shrunk headline and never show an inflated title for
    # a low-sample player. In scratch view this equals the headline; in handicap
    # view it is the shrunk scratch rating (the "Scratch_Title" skill indicator).
    if prior_sessions > 0 and scratch_anchor is not None:
        ps_lit_s = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
        sanchor_lit = f"CAST({float(scratch_anchor)!r} AS DOUBLE)"
        published_scratch_expr = (
            f"CAST(ROUND(LEAST(GREATEST("
            f"(CAST(games_played AS DOUBLE) * CAST(scratch_elo AS DOUBLE) "
            f"+ {ps_lit_s} * {sanchor_lit}) "
            f"/ NULLIF(CAST(games_played AS DOUBLE) + {ps_lit_s}, 0)"
            f", 0), 3500), 0) AS INTEGER)"
        )
    else:
        published_scratch_expr = "CAST(ROUND(LEAST(GREATEST(scratch_elo, 0), 3500), 0) AS INTEGER)"

    title_col = f""",
            CASE 
                WHEN published_scratch_int >= 2600 THEN 'SGM'
                WHEN published_scratch_int >= 2500 THEN 'GM'
                WHEN published_scratch_int >= 2400 THEN 'IM'
                WHEN published_scratch_int >= 2300 THEN 'FM'
                WHEN published_scratch_int >= 2200 THEN 'CM'
                WHEN published_scratch_int >= 2000 THEN 'Expert'
                WHEN published_scratch_int >= 1800 THEN 'Advanced'
                WHEN published_scratch_int >= 1600 THEN 'Intermediate'
                WHEN published_scratch_int >= 1400 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""

    query = f"""
        WITH filtered AS (
            SELECT *
            FROM players_df
            WHERE games_played >= {min_games}
        ),
        ranked AS (
            SELECT
                *,
                CAST(ROUND(LEAST(GREATEST(elo_rating, 0), 3500), 0) AS INTEGER) AS raw_elo_int,
                {published_expr} AS published_elo_int,
                {published_scratch_expr} AS published_scratch_int
            FROM filtered
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY published_elo_int DESC, games_played DESC, player_name ASC, player_id ASC) AS INTEGER) AS Rank,
            published_elo_int AS {elo_col_name},
            raw_elo_int AS {elo_col_name}_Raw{title_col},
            player_id AS Player_ID,
            player_name AS Player_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games,
            CAST(provisional_games AS INTEGER) AS Provisional_Games
        FROM ranked
        ORDER BY Rank ASC
        LIMIT {top_n}
    """

    result = duckdb.sql(query).pl()
    return result, query, prior_anchor


def show_top_pairs(
    results_df: pl.DataFrame,
    top_n: int,
    min_games: int = 5,
    use_handicap: bool = False,
    players_df: Optional[pl.DataFrame] = None,
    prior_sessions: int = 0,
) -> Tuple[pl.DataFrame, str, Optional[float]]:
    """Get top pairs sorted by Elo rating using SQL.

    When ``prior_sessions > 0`` and a prior anchor (median pair Elo of the
    qualifying subset) is available, the headline pair Elo is the
    Bayesian-shrunk "Published" Elo, computed the same way as for players in
    :func:`show_top_players`. Both Published and Raw are returned in the
    output; the leaderboard is ordered by Published. Returns
    ``(df, sql, prior_anchor)``.
    """
    if results_df.is_empty():
        return results_df, "", None

    elo_col = "handicap_pair_elo" if use_handicap else "scratch_pair_elo"
    pct_col = "COALESCE(handicap_percentage, scratch_percentage)" if use_handicap else "scratch_percentage"

    pair_elo_col_name = "HC_Pair_Elo" if use_handicap else "Pair_Elo"
    title_col_name = "Scratch_Title" if use_handicap else "Title"

    anchor_query = f"""
        WITH pair_anchor AS (
            SELECT
                pair_id,
                ARG_MAX({elo_col}, date) AS avg_pair_elo,
                COUNT(*) AS games_played
            FROM results_df
            GROUP BY pair_id
        )
        SELECT MEDIAN(avg_pair_elo) AS anchor
        FROM pair_anchor
        WHERE games_played >= {min_games}
    """
    anchor_df = duckdb.sql(anchor_query).pl()
    if anchor_df.is_empty():
        prior_anchor: Optional[float] = None
    else:
        val = anchor_df.item(0, 0)
        prior_anchor = float(val) if val is not None else None

    if prior_sessions > 0 and prior_anchor is not None:
        ps_lit = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
        anchor_lit = f"CAST({float(prior_anchor)!r} AS DOUBLE)"
        published_expr = (
            f"CAST(ROUND(LEAST(GREATEST("
            f"(CAST(games_played AS DOUBLE) * CAST(avg_pair_elo AS DOUBLE) "
            f"+ {ps_lit} * {anchor_lit}) "
            f"/ NULLIF(CAST(games_played AS DOUBLE) + {ps_lit}, 0)"
            f", 0), 3500), 0) AS INTEGER)"
        )
    else:
        published_expr = "CAST(ROUND(LEAST(GREATEST(avg_pair_elo, 0), 3500), 0) AS INTEGER)"
    
    # Build Title column - use lower title of the two players based on their
    # *published* (Bayesian-shrunk) scratch Elo, so a pair never inherits an
    # inflated title from a low-sample partner. Each player's scratch Elo is
    # shrunk toward the population scratch median using their own games count,
    # mirroring the headline shrinkage.
    scratch_anchor: Optional[float] = None
    if players_df is not None and not players_df.is_empty():
        sa_df = duckdb.sql(
            f"""
            SELECT MEDIAN(scratch_elo) AS scratch_anchor
            FROM players_df
            WHERE games_played >= {min_games}
            """
        ).pl()
        if not sa_df.is_empty():
            sv = sa_df.item(0, 0)
            scratch_anchor = float(sv) if sv is not None else None

    def _pub_scratch_sql(alias: str) -> str:
        """Shrunk, chess-clamped scratch Elo for a joined player alias."""
        if prior_sessions > 0 and scratch_anchor is not None:
            ps_lit_s = f"CAST({float(prior_sessions)!r} AS DOUBLE)"
            sanchor_lit = f"CAST({float(scratch_anchor)!r} AS DOUBLE)"
            return (
                f"CAST(ROUND(LEAST(GREATEST("
                f"(CAST(COALESCE({alias}.games_played, 0) AS DOUBLE) "
                f"* CAST(COALESCE({alias}.scratch_elo, 0) AS DOUBLE) "
                f"+ {ps_lit_s} * {sanchor_lit}) "
                f"/ NULLIF(CAST(COALESCE({alias}.games_played, 0) AS DOUBLE) + {ps_lit_s}, 0)"
                f", 0), 3500), 0) AS INTEGER)"
            )
        return f"CAST(ROUND(LEAST(GREATEST(COALESCE({alias}.scratch_elo, 0), 0), 3500), 0) AS INTEGER)"

    if players_df is not None and not players_df.is_empty():
        p1_pub = _pub_scratch_sql("p1")
        p2_pub = _pub_scratch_sql("p2")
        # Join with players_df to get individual player scratch Elo and calculate lower title
        title_col = """,
            CASE 
                -- Calculate title rank for player1 (1=SGM, 10=Beginner)
                WHEN p1_pub_scratch >= 2600 THEN 1
                WHEN p1_pub_scratch >= 2500 THEN 2
                WHEN p1_pub_scratch >= 2400 THEN 3
                WHEN p1_pub_scratch >= 2300 THEN 4
                WHEN p1_pub_scratch >= 2200 THEN 5
                WHEN p1_pub_scratch >= 2000 THEN 6
                WHEN p1_pub_scratch >= 1800 THEN 7
                WHEN p1_pub_scratch >= 1600 THEN 8
                WHEN p1_pub_scratch >= 1400 THEN 9
                ELSE 10
            END AS p1_title_rank,
            CASE 
                -- Calculate title rank for player2 (1=SGM, 10=Beginner)
                WHEN p2_pub_scratch >= 2600 THEN 1
                WHEN p2_pub_scratch >= 2500 THEN 2
                WHEN p2_pub_scratch >= 2400 THEN 3
                WHEN p2_pub_scratch >= 2300 THEN 4
                WHEN p2_pub_scratch >= 2200 THEN 5
                WHEN p2_pub_scratch >= 2000 THEN 6
                WHEN p2_pub_scratch >= 1800 THEN 7
                WHEN p2_pub_scratch >= 1600 THEN 8
                WHEN p2_pub_scratch >= 1400 THEN 9
                ELSE 10
            END AS p2_title_rank"""
        
        title_select = f""",
            CASE 
                -- Use GREATEST to get the higher rank number, which corresponds to the lower title
                -- (Higher rank number = lower title: 1=SGM, 2=GM, ..., 10=Beginner)
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 1 THEN 'SGM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 2 THEN 'GM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 3 THEN 'IM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 4 THEN 'FM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 5 THEN 'CM'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 6 THEN 'Expert'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 7 THEN 'Advanced'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 8 THEN 'Intermediate'
                WHEN GREATEST(COALESCE(p1_title_rank, 10), COALESCE(p2_title_rank, 10)) = 9 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""
    else:
        # Fallback: use pair scratch Elo if players_df not available
        title_col = f""",
            CASE 
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2600 THEN 'SGM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2500 THEN 'GM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2400 THEN 'IM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2300 THEN 'FM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2200 THEN 'CM'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 2000 THEN 'Expert'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 1800 THEN 'Advanced'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 1600 THEN 'Intermediate'
                WHEN CAST(ROUND(LEAST(GREATEST(avg_scratch_elo, 0), 3500), 0) AS INTEGER) >= 1400 THEN 'Novice'
                ELSE 'Beginner'
            END AS {title_col_name}"""
        title_select = ""
    
    query = f"""
        WITH pair_stats AS (
            SELECT 
                pair_id,
                ARG_MAX(pair_name, date) AS pair_name,
                ARG_MAX(player1_id, date) AS player1_id,
                ARG_MAX(player2_id, date) AS player2_id,
                -- Headline Elo uses Latest semantics (ARG_MAX over date) to
                -- match show_top_players, which already uses ARG_MAX. Avoids
                -- the early-tournament-lock-in bias that ACBL's AVG method
                -- exposed: one lucky early session permanently inflating the
                -- AVG even after later results regress to the pair's real
                -- skill level. The percentage / IV / stdev aggregates below
                -- intentionally stay as AVG because users expect "average
                -- across all my tournaments" for those.
                ARG_MAX(scratch_pair_elo, date) AS avg_scratch_elo,
                ARG_MAX(COALESCE(handicap_pair_elo, scratch_pair_elo), date) AS avg_handicap_elo,
                ARG_MAX({elo_col}, date) AS avg_pair_elo,
                AVG(scratch_percentage) AS avg_scratch_pct,
                AVG(COALESCE(handicap_percentage, scratch_percentage)) AS avg_handicap_pct,
                AVG(iv_bonus) AS avg_iv_bonus,
                AVG({pct_col}) AS avg_percentage,
                STDDEV_SAMP({pct_col}) AS stdev_percentage,
                COUNT(*) AS games_played,
                SUM(CASE WHEN score_status = 'provisional' THEN 1 ELSE 0 END)
                    AS provisional_games
            FROM results_df
            GROUP BY pair_id
        ),
        filtered AS (
            SELECT *
            FROM pair_stats
            WHERE games_played >= {min_games}
        ),
        ranked AS (
            SELECT
                *,
                CAST(ROUND(LEAST(GREATEST(avg_pair_elo, 0), 3500), 0) AS INTEGER) AS raw_pair_elo_int,
                {published_expr} AS published_pair_elo_int
            FROM filtered
        )"""

    if players_df is not None and not players_df.is_empty():
        query += f""",
        with_player_scratch AS (
            SELECT 
                f.*,
                {p1_pub} AS p1_pub_scratch,
                {p2_pub} AS p2_pub_scratch
            FROM ranked f
            LEFT JOIN players_df p1 ON f.player1_id = p1.player_id
            LEFT JOIN players_df p2 ON f.player2_id = p2.player_id
        ),
        with_player_titles AS (
            SELECT 
                w.*{title_col}
            FROM with_player_scratch w
        )
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY published_pair_elo_int DESC, games_played DESC, pair_name ASC, pair_id ASC) AS INTEGER) AS Rank,
            published_pair_elo_int AS {pair_elo_col_name},
            raw_pair_elo_int AS {pair_elo_col_name}_Raw{title_select},
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games,
            CAST(provisional_games AS INTEGER) AS Provisional_Games
        FROM with_player_titles
        ORDER BY Rank ASC
        LIMIT {top_n}
    """
    else:
        query += f"""
        SELECT 
            CAST(ROW_NUMBER() OVER (ORDER BY published_pair_elo_int DESC, games_played DESC, pair_name ASC, pair_id ASC) AS INTEGER) AS Rank,
            published_pair_elo_int AS {pair_elo_col_name},
            raw_pair_elo_int AS {pair_elo_col_name}_Raw{title_col},
            pair_id AS Pair_ID,
            pair_name AS Pair_Name,
            ROUND(avg_scratch_pct, 1) AS Avg_Scratch,
            ROUND(avg_handicap_pct, 1) AS Avg_Handicap,
            ROUND(avg_iv_bonus, 1) AS Avg_IV_Bonus,
            ROUND(stdev_percentage, 1) AS Pct_Stdev,
            CAST(games_played AS INTEGER) AS Games,
            CAST(provisional_games AS INTEGER) AS Provisional_Games
        FROM ranked
        ORDER BY Rank ASC
        LIMIT {top_n}
    """

    result = duckdb.sql(query).pl()
    return result, query, prior_anchor


# -------------------------------
# One-call report (used by the MCP server)
# -------------------------------
def run_leaderboard_report(
    *,
    rating: str = "Players",
    score: str = "Scratch",
    top_n: int = DEFAULT_TOP_N,
    min_games: int = DEFAULT_MIN_GAMES,
    prior_sessions: int = DEFAULT_PRIOR_SESSIONS,
    series_id: Optional[int | str] = None,
    tournament: Optional[str] = None,
    tournament_contains: Optional[str] = None,
    club: Optional[str] = None,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    api_key: Optional[str] = None,
    fetch_iv: bool = True,
) -> Dict[str, Any]:
    """Full pipeline: load parquet -> filter -> aggregate -> leaderboard SQL.

    Mirrors the Streamlit app's report path (percentage filter, sidebar
    filters, ARG_MAX aggregation, Bayesian shrinkage). Explicit
    ``date_from``/``date_to`` override a named ``date_range``.
    """
    if rating not in ("Players", "Pairs"):
        raise ValueError(f"rating must be 'Players' or 'Pairs', got {rating!r}")
    if score not in ("Scratch", "Handicap"):
        raise ValueError(f"score must be 'Scratch' or 'Handicap', got {score!r}")
    use_handicap = score == "Handicap"

    if date_range and not (date_from or date_to):
        if date_range not in DATE_RANGE_OPTIONS:
            raise ValueError(
                f"Unknown date_range {date_range!r}; valid: {list(DATE_RANGE_OPTIONS)}"
            )
        date_from, date_to = date_range_bounds(date_range)

    normalized_series_id = resolve_series_id(series_id)
    results_df, meta = load_results(api_key, fetch_iv)
    results_df = filter_valid_percentages(results_df)
    results_df = filter_results(
        results_df,
        series_id=normalized_series_id,
        tournament=tournament,
        tournament_contains=tournament_contains,
        club=club,
        date_from=date_from,
        date_to=date_to,
    )
    provenance = score_provenance_counts(results_df)
    results_df = filter_score_available(results_df, use_handicap)

    if rating == "Players":
        players_df = aggregate_players_from_results(results_df, use_handicap)
        table, sql, prior_anchor = show_top_players(
            players_df, top_n, min_games,
            use_handicap=use_handicap, prior_sessions=prior_sessions,
        )
    else:
        table, sql, prior_anchor = show_top_pairs(
            results_df, top_n, min_games,
            use_handicap=use_handicap, players_df=None,
            prior_sessions=prior_sessions,
        )
    del sql

    name_token = (player_name or "").strip()
    number_token = (player_number or "").strip()
    table = filter_ffbridge_leaderboard(
        table,
        rating_type=rating,
        player_name=name_token,
        player_number=number_token,
    )

    return {
        "rows": table.to_dicts() if not table.is_empty() else [],
        "rating": rating,
        "score": score,
        "top_n": top_n,
        "min_games": min_games,
        "prior_sessions": prior_sessions,
        "prior_anchor": prior_anchor,
        "series_id": normalized_series_id,
        "series_name": (
            None if normalized_series_id is None else SERIES_NAMES[normalized_series_id]
        ),
        "tournament": (tournament or "").strip() or None,
        "tournament_contains": (tournament_contains or "").strip() or None,
        "club": club,
        "player_name": name_token or None,
        "player_number": number_token or None,
        "date_from": date_from,
        "date_to": date_to,
        "filtered_result_rows": results_df.height,
        "dataset_built_at": meta.get("built_at"),
        "score_provenance": provenance,
    }


def run_player_history(
    player_id: str,
    *,
    limit: int = 100,
    api_key: Optional[str] = None,
    fetch_iv: bool = True,
) -> Dict[str, Any]:
    """Return one player's persisted per-session history."""
    pid = str(player_id).strip()
    if not pid.isdigit():
        raise ValueError("player_id must contain digits only")
    results_df, meta = load_results(api_key, fetch_iv)
    results_df = filter_valid_percentages(results_df)
    player_expr = (
        (pl.col("player1_id").cast(pl.Utf8) == pid)
        | (pl.col("player2_id").cast(pl.Utf8) == pid)
    )
    all_sessions = results_df.filter(player_expr)
    wanted = [
        "date", "tournament_id", "club_name", "pair_id", "pair_name",
        "player1_id", "player1_name", "player2_id", "player2_name",
        "scratch_percentage", "handicap_percentage", "iv_bonus", "rank",
        "score_source", "score_status", "scratch_score_status",
        "handicap_score_status", "score_source_url",
        "player1_scratch_elo_after", "player2_scratch_elo_after",
        "player1_handicap_elo_after", "player2_handicap_elo_after",
    ]
    sessions = (
        all_sessions.sort("date", descending=True)
        .select([column for column in wanted if column in all_sessions.columns])
        .head(limit)
    )
    return {
        "player_id": pid,
        "sessions": sessions.to_dicts(),
        "total_sessions": all_sessions.height,
        "dataset_built_at": meta.get("built_at"),
    }
