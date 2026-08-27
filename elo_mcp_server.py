"""MCP server exposing FFBridge and ACBL Elo rating reports.

Transport: streamable HTTP (endpoint /mcp) on ELO_MCP_PORT (default 8510),
stateless with JSON responses so plain HTTP clients and cloudflared work
without session affinity.

Every tool is an HTTP client of a first-party Elo REST API. The MCP process
does not import report libraries, read parquet files, or call third-party APIs.

Deployment: elo-mcp container, started by ../7nt/elo_ratings_start.ps1.
GET /health is used by the wslc watchdog and deploy health checks.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

ACBL_API_BASE_URL = os.environ.get("ACBL_API_BASE_URL", "http://localhost:8505").rstrip("/")
FFBRIDGE_API_BASE_URL = os.environ.get(
    "FFBRIDGE_API_BASE_URL", "http://localhost:8511"
).rstrip("/")
ELO_MCP_PORT = int(os.environ.get("ELO_MCP_PORT", "8510"))
MCP_SCHEMA_VERSION = "2026-08-25-tournament-v2"

# Payload cap: MCP responses are JSON over a single HTTP exchange; thousands of
# rows are fine, tens of thousands are not.
_MAX_TOP_N = 5000
_MAX_HISTORY_ROWS = 500
_API_TIMEOUT_S = 300

mcp = MCPServer("elo-ratings", version=MCP_SCHEMA_VERSION)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for the wslc watchdog / deploy health check."""
    return JSONResponse(
        {
            "status": "ok",
            "service": "elo-mcp",
            "schema_version": MCP_SCHEMA_VERSION,
            "acbl_api_base_url": ACBL_API_BASE_URL,
            "ffbridge_api_base_url": FFBRIDGE_API_BASE_URL,
        }
    )


# -------------------------------
# First-party API clients
# -------------------------------
def _api_get(base_url: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(
        f"{base_url}{path}",
        params={key: value for key, value in params.items() if value is not None},
        timeout=_API_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def _ffbridge_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return _api_get(FFBRIDGE_API_BASE_URL, path, params)


def _acbl_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return _api_get(ACBL_API_BASE_URL, path, params)


@mcp.tool()
def ffbridge_dataset_info(api_backend: Optional[str] = None) -> Dict[str, Any]:
    """Summary of the FFBridge Elo dataset: build time, result-row count,
    covered date range, available club names (for the club filter of the
    leaderboard tools), result-link coverage, quality status, and valid
    date_range choices."""
    return _ffbridge_get(
        "/ffbridge/dataset-info", {"api_backend": api_backend}
    )


@mcp.tool()
def ffbridge_top_players(
    top_n: int = 250,
    min_games: int = 10,
    score: str = "Scratch",
    club: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    prior_sessions: int = 50,
    api_backend: Optional[str] = None,
    series_id: Optional[str] = None,
    tournament_name: Optional[str] = None,
    tournament: Optional[str] = None,
    tournament_contains: Optional[str] = None,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
) -> Dict[str, Any]:
    """FFBridge player Elo leaderboard (same pipeline as ffbridge-elo.7nt.info).

    tournament is an exact event-name match; tournament_contains is a
    case-insensitive substring match. They filter result rows before Elo is
    calculated and cannot be combined with series_id/tournament_name.
    club and player_name are typo-tolerant; numeric player IDs remain exact.
    date_range is a named window (e.g. 'Current FFBridge year', 'Last 1 year');
    explicit date_from/date_to (YYYY-MM-DD, inclusive) override it.
    When deployed quality sidecars are available, rows include DD/par/sacrifice
    quality metrics. The response reports quality and result-link status.
    """
    return _ffbridge_get(
        "/ffbridge/report",
        {
            "rating_type": "Players",
            "score": score,
            "top_n": min(top_n, _MAX_TOP_N),
            "min_games": min_games,
            "prior_sessions": prior_sessions,
            "api_backend": api_backend,
            "series_id": series_id,
            "tournament_name": tournament_name,
            "tournament": tournament,
            "tournament_contains": tournament_contains,
            "club": club,
            "player_name": player_name,
            "player_number": player_number,
            "date_range": date_range,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


@mcp.tool()
def ffbridge_top_pairs(
    top_n: int = 250,
    min_games: int = 10,
    score: str = "Scratch",
    club: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    prior_sessions: int = 50,
    api_backend: Optional[str] = None,
    series_id: Optional[str] = None,
    tournament_name: Optional[str] = None,
    tournament: Optional[str] = None,
    tournament_contains: Optional[str] = None,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
) -> Dict[str, Any]:
    """FFBridge pair Elo leaderboard. Parameters as in ffbridge_top_players.
    Pair Elo uses Latest semantics (rating after the pair's most recent
    session), shrunk toward the qualifying-pair median when prior_sessions > 0.
    Deployed quality sidecars add DD/par/sacrifice metrics and ranks.
    """
    return _ffbridge_get(
        "/ffbridge/report",
        {
            "rating_type": "Pairs",
            "score": score,
            "top_n": min(top_n, _MAX_TOP_N),
            "min_games": min_games,
            "prior_sessions": prior_sessions,
            "api_backend": api_backend,
            "series_id": series_id,
            "tournament_name": tournament_name,
            "tournament": tournament,
            "tournament_contains": tournament_contains,
            "club": club,
            "player_name": player_name,
            "player_number": player_number,
            "date_range": date_range,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


@mcp.tool()
def ffbridge_top_players_v2(
    top_n: int = 250,
    min_games: int = 10,
    score: str = "Scratch",
    club: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    prior_sessions: int = 50,
    api_backend: Optional[str] = None,
    series_id: Optional[str] = None,
    tournament_name: Optional[str] = None,
    tournament: Optional[str] = None,
    tournament_contains: Optional[str] = None,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Versioned FFBridge player leaderboard schema.

    tournament is an exact event-name match; tournament_contains is a
    case- and accent-insensitive substring applied before Elo calculation.
    """
    return ffbridge_top_players(
        top_n=top_n,
        min_games=min_games,
        score=score,
        club=club,
        date_range=date_range,
        date_from=date_from,
        date_to=date_to,
        prior_sessions=prior_sessions,
        api_backend=api_backend,
        series_id=series_id,
        tournament_name=tournament_name,
        tournament=tournament,
        tournament_contains=tournament_contains,
        player_name=player_name,
        player_number=player_number,
    )


@mcp.tool()
def ffbridge_top_pairs_v2(
    top_n: int = 250,
    min_games: int = 10,
    score: str = "Scratch",
    club: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    prior_sessions: int = 50,
    api_backend: Optional[str] = None,
    series_id: Optional[str] = None,
    tournament_name: Optional[str] = None,
    tournament: Optional[str] = None,
    tournament_contains: Optional[str] = None,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Versioned FFBridge pair leaderboard schema with tournament filters."""
    return ffbridge_top_pairs(
        top_n=top_n,
        min_games=min_games,
        score=score,
        club=club,
        date_range=date_range,
        date_from=date_from,
        date_to=date_to,
        prior_sessions=prior_sessions,
        api_backend=api_backend,
        series_id=series_id,
        tournament_name=tournament_name,
        tournament=tournament,
        tournament_contains=tournament_contains,
        player_name=player_name,
        player_number=player_number,
    )


@mcp.tool()
def ffbridge_tournaments(
    club: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    contains: Optional[str] = None,
    limit: int = 500,
    api_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover canonical FFBridge tournament names.

    Optional club/date filters narrow the source result rows. contains is a
    case-insensitive substring filter on tournament names.
    """
    return _ffbridge_get(
        "/ffbridge/tournaments",
        {
            "club": club,
            "date_from": date_from,
            "date_to": date_to,
            "contains": contains,
            "limit": min(limit, 5000),
            "api_backend": api_backend,
        },
    )


@mcp.tool()
def ffbridge_player_history(
    player_id: str,
    limit: int = 100,
    api_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-session tournament history for one FFBridge player (newest first):
    date, tournament, club, partner/pair, scratch and handicap percentages,
    national rank, Elo after the session, and Results_URL, plus link-coverage
    status."""
    return _ffbridge_get(
        "/ffbridge/player-history",
        {
            "player_id": player_id,
            "limit": min(limit, _MAX_HISTORY_ROWS),
            "api_backend": api_backend,
        },
    )


# -------------------------------
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
    moving_avg_days: Optional[int] = None,
    elo_rating_type: Optional[str] = None,
    date_range: Optional[str] = None,
    date_from: Optional[str] = None,
    online_filter: Optional[str] = None,
    strata: Optional[str] = None,
    prior_sessions: Optional[int] = None,
    min_skill_z: Optional[float] = None,
    player_name: Optional[str] = None,
    player_number: Optional[str] = None,
    masterpoints_range: Optional[str] = None,
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
            "moving_avg_days": moving_avg_days,
            "elo_rating_type": elo_rating_type,
            "date_range": date_range,
            "date_from": date_from,
            "online_filter": online_filter,
            "strata": strata,
            "prior_sessions": prior_sessions,
            "min_skill_z": min_skill_z,
            "player_name": player_name,
            "player_number": player_number,
            "masterpoints_range": masterpoints_range,
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
    """Board detail for one ACBL player or pair from acbl-api /acbl/detail.

    For rating_type='Players' pass player_id (ACBL number); for 'Pairs' pass
    pair_ids as 'id1-id2'. Rows include Results_URL and the response includes
    results_url_status. Other params as in acbl_report.
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
        f"[elo-mcp] start {datetime.now(timezone.utc).isoformat()} "
        f"on :{ELO_MCP_PORT} (endpoint /mcp, health /health); "
        f"ACBL API -> {ACBL_API_BASE_URL}; FFBridge API -> {FFBRIDGE_API_BASE_URL}",
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
