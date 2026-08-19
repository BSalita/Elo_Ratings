"""MCP server exposing FFBridge and ACBL Elo rating reports.

Transport: streamable HTTP (endpoint /mcp) on ELO_MCP_PORT (default 8510),
stateless with JSON responses so plain HTTP clients and cloudflared work
without session affinity.

FFBridge tools run the shared report pipeline in ffbridge_report_service
against the persisted Elo parquet set (FFBRIDGE_CACHE_DIR). ACBL tools proxy
the acbl-api FastAPI service (ACBL_API_BASE_URL, default http://localhost:8505)
so its multi-GB in-memory frames stay in that one process.

Deployment: elo-mcp container, started by ../7nt/elo_ratings_start.ps1.
GET /health is used by the wslc watchdog and deploy health checks.
"""

import os
from typing import Any, Dict, Optional

import polars as pl
import requests
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

import ffbridge_report_service as ffsvc

ACBL_API_BASE_URL = os.environ.get("ACBL_API_BASE_URL", "http://localhost:8505").rstrip("/")
ELO_MCP_PORT = int(os.environ.get("ELO_MCP_PORT", "8510"))

# Payload cap: MCP responses are JSON over a single HTTP exchange; thousands of
# rows are fine, tens of thousands are not.
_MAX_TOP_N = 2000
_MAX_HISTORY_ROWS = 500
_ACBL_TIMEOUT_S = 300

mcp = MCPServer("elo-ratings")


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for the wslc watchdog / deploy health check."""
    key = ffsvc.resolve_elo_cache_key(ffsvc.default_api_key(), True)
    return JSONResponse(
        {
            "status": "ok",
            "service": "elo-mcp",
            "ffbridge_parquet_key": key,
            "acbl_api_base_url": ACBL_API_BASE_URL,
        }
    )


# -------------------------------
# FFBridge tools (shared pipeline, persisted parquet)
# -------------------------------
@mcp.tool()
def ffbridge_dataset_info() -> Dict[str, Any]:
    """Summary of the FFBridge Elo dataset: build time, result-row count,
    covered date range, available club names (for the club filter of the
    leaderboard tools), and valid date_range choices."""
    return ffsvc.dataset_info()


@mcp.tool()
def ffbridge_top_players(
    top_n: int = 50,
    min_games: int = ffsvc.DEFAULT_MIN_GAMES,
    score: str = "Scratch",
    club: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    prior_sessions: int = ffsvc.DEFAULT_PRIOR_SESSIONS,
) -> Dict[str, Any]:
    """FFBridge player Elo leaderboard (same pipeline as ffbridge-elo.7nt.info).

    score: 'Scratch' or 'Handicap'. club: exact club name from
    ffbridge_dataset_info. date_range: named window (e.g. 'Current FFBridge
    year', 'Last 1 year'); explicit date_from/date_to (YYYY-MM-DD, inclusive)
    override it. prior_sessions: Bayesian shrinkage weight toward the
    qualifying-population median (0 disables; headline equals raw Elo).
    """
    return ffsvc.run_leaderboard_report(
        rating="Players",
        score=score,
        top_n=min(top_n, _MAX_TOP_N),
        min_games=min_games,
        prior_sessions=prior_sessions,
        club=club,
        date_range=date_range,
        date_from=date_from,
        date_to=date_to,
    )


@mcp.tool()
def ffbridge_top_pairs(
    top_n: int = 50,
    min_games: int = ffsvc.DEFAULT_MIN_GAMES,
    score: str = "Scratch",
    club: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    prior_sessions: int = ffsvc.DEFAULT_PRIOR_SESSIONS,
) -> Dict[str, Any]:
    """FFBridge pair Elo leaderboard. Parameters as in ffbridge_top_players.
    Pair Elo uses Latest semantics (rating after the pair's most recent
    session), shrunk toward the qualifying-pair median when prior_sessions > 0.
    """
    return ffsvc.run_leaderboard_report(
        rating="Pairs",
        score=score,
        top_n=min(top_n, _MAX_TOP_N),
        min_games=min_games,
        prior_sessions=prior_sessions,
        club=club,
        date_range=date_range,
        date_from=date_from,
        date_to=date_to,
    )


@mcp.tool()
def ffbridge_player_history(
    player_id: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """Per-session tournament history for one FFBridge player (newest first):
    date, tournament, club, partner/pair, scratch and handicap percentages,
    rank, and Elo after the session."""
    limit = min(limit, _MAX_HISTORY_ROWS)
    results_df, meta = ffsvc.load_results()
    results_df = ffsvc.filter_valid_percentages(results_df)
    pid = str(player_id)
    df = results_df.filter(
        (pl.col("player1_id").cast(pl.Utf8) == pid)
        | (pl.col("player2_id").cast(pl.Utf8) == pid)
    ).sort("date", descending=True)
    wanted = [
        "date", "tournament_id", "club_name", "pair_id", "pair_name",
        "player1_id", "player1_name", "player2_id", "player2_name",
        "scratch_percentage", "handicap_percentage", "iv_bonus", "rank",
        "player1_scratch_elo_after", "player2_scratch_elo_after",
        "player1_handicap_elo_after", "player2_handicap_elo_after",
    ]
    df = df.select([c for c in wanted if c in df.columns]).head(limit)
    return {
        "player_id": pid,
        "sessions": df.to_dicts(),
        "total_sessions": results_df.filter(
            (pl.col("player1_id").cast(pl.Utf8) == pid)
            | (pl.col("player2_id").cast(pl.Utf8) == pid)
        ).height,
        "dataset_built_at": meta.get("built_at"),
    }


# -------------------------------
# ACBL tools (proxy to acbl-api FastAPI)
# -------------------------------
def _acbl_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(
        f"{ACBL_API_BASE_URL}{path}",
        params={k: v for k, v in params.items() if v is not None},
        timeout=_ACBL_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def acbl_health() -> Dict[str, Any]:
    """Health and memory status of the ACBL Elo API server (acbl-api)."""
    return _acbl_get("/health", {})


@mcp.tool()
def acbl_report(
    club_or_tournament: str = "club",
    rating_type: str = "Players",
    top_n: int = 100,
    min_sessions: int = 10,
    rating_method: Optional[str] = None,
    elo_rating_type: Optional[str] = None,
    date_from: Optional[str] = None,
    online_filter: Optional[str] = None,
    strata: Optional[str] = None,
    prior_sessions: Optional[int] = None,
    min_skill_z: Optional[float] = None,
) -> Dict[str, Any]:
    """ACBL Elo leaderboard (players or pairs) from acbl-api /acbl/report.

    club_or_tournament: 'club' or 'tournament'. rating_type: 'Players' or
    'Pairs'. Optional params fall back to the API server defaults:
    rating_method ('Latest'/'Avg'/'Max'/'Moving Avg'), date_from (ISO date),
    online_filter ('All'/'Local Only'/'Online Only'), strata (event MP-limit
    bucket), prior_sessions (Bayesian shrinkage weight, 0 disables),
    min_skill_z (elite skill gate; <= -90 disables).
    """
    return _acbl_get(
        "/acbl/report",
        {
            "club_or_tournament": club_or_tournament,
            "rating_type": rating_type,
            "top_n": min(top_n, _MAX_TOP_N),
            "min_sessions": min_sessions,
            "rating_method": rating_method,
            "elo_rating_type": elo_rating_type,
            "date_from": date_from,
            "online_filter": online_filter,
            "strata": strata,
            "prior_sessions": prior_sessions,
            "min_skill_z": min_skill_z,
        },
    )


@mcp.tool()
def acbl_detail(
    club_or_tournament: str = "club",
    rating_type: str = "Players",
    player_id: Optional[str] = None,
    pair_ids: Optional[str] = None,
    elo_rating_type: Optional[str] = None,
    date_from: Optional[str] = None,
    online_filter: Optional[str] = None,
    strata: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-session detail for one ACBL player or pair from acbl-api /acbl/detail.

    For rating_type='Players' pass player_id (ACBL number); for 'Pairs' pass
    pair_ids as 'id1-id2'. Other params as in acbl_report.
    """
    return _acbl_get(
        "/acbl/detail",
        {
            "club_or_tournament": club_or_tournament,
            "rating_type": rating_type,
            "player_id": player_id,
            "pair_ids": pair_ids,
            "elo_rating_type": elo_rating_type,
            "date_from": date_from,
            "online_filter": online_filter,
            "strata": strata,
        },
    )


if __name__ == "__main__":
    print(
        f"[elo-mcp] starting on :{ELO_MCP_PORT} (endpoint /mcp, health /health); "
        f"ACBL proxy -> {ACBL_API_BASE_URL}",
        flush=True,
    )
    # Stateless + JSON responses: plain request/response tools, no session
    # affinity needed behind cloudflared, and curl-testable.
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=ELO_MCP_PORT,
        stateless_http=True,
        json_response=True,
    )
