"""First-party REST API for persisted FFBridge Elo reports."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query

import ffbridge_board_service as boards
import ffbridge_report_service as reports
import ffbridge_session_ranking_service as rankings
from streamlitlib.memory_usage import get_memory_usage_dict


FFBRIDGE_API_BUILD_TAG = "2026-09-08-quality-fastpath"
app = FastAPI(title="FFBridge Elo API", version="1.5.0")


def _run(callable_, /, **kwargs):
    try:
        return callable_(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _warm_dataset() -> None:
    try:
        reports.load_results()
        print("[ffbridge-api] dataset warmup done", flush=True)
    except Exception as exc:
        print(f"[ffbridge-api] dataset warmup failed: {exc}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_warm_dataset, name="ffbridge-api-warmup", daemon=True).start()


@app.get("/health")
def health() -> dict:
    """Liveness only. Do not load parquets here — that OOMs the shared container
    and makes MortyBridgeBot's 2s health probe time out while Streamlit stays up.
    Dataset details live on /ffbridge/dataset-info.
    """
    return {
        "status": "ok",
        "service": "ffbridge-api",
        "api_version": app.version,
        "build_tag": FFBRIDGE_API_BUILD_TAG,
        "memory": get_memory_usage_dict(),
    }


@app.get("/ffbridge/dataset-info")
def dataset_info(
    api_backend: str | None = Query(None),
) -> dict:
    return _run(reports.dataset_info, api_key=api_backend)


@app.get("/ffbridge/report")
def leaderboard_report(
    rating_type: str = Query("Players", pattern="^(Players|Pairs)$"),
    score: str = Query("Scratch", pattern="^(Scratch|Handicap)$"),
    top_n: int = Query(reports.DEFAULT_TOP_N, ge=1, le=5000),
    min_games: int = Query(reports.DEFAULT_MIN_GAMES, ge=1, le=10000),
    prior_sessions: int = Query(
        reports.DEFAULT_PRIOR_SESSIONS, ge=0, le=1000
    ),
    api_backend: str | None = Query(None),
    series_id: str | None = Query(None),
    tournament_name: str | None = Query(None),
    tournament: str | None = Query(None),
    tournament_contains: str | None = Query(None),
    club: str | None = Query(None),
    player_name: str | None = Query(None),
    player_number: str | None = Query(None, pattern=r"^\d*$"),
    date_range: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
) -> dict:
    """Return filtered Elo rows with role-aware bridge-quality metrics."""
    tournament_filters = [
        value
        for value in (
            series_id,
            tournament_name,
            tournament,
            tournament_contains,
        )
        if value
    ]
    if len(tournament_filters) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Pass only one of series_id, tournament_name, tournament, "
                "or tournament_contains"
            ),
        )
    return _run(
        reports.run_leaderboard_report,
        rating=rating_type,
        score=score,
        top_n=top_n,
        min_games=min_games,
        prior_sessions=prior_sessions,
        api_key=api_backend,
        series_id=series_id or tournament_name,
        tournament=tournament,
        tournament_contains=tournament_contains,
        club=club,
        player_name=player_name,
        player_number=player_number,
        date_range=date_range,
        date_from=date_from,
        date_to=date_to,
    )


@app.get("/ffbridge/tournaments")
def tournaments(
    club: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    contains: str | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    api_backend: str | None = Query(None),
) -> dict:
    return _run(
        reports.list_tournaments,
        club=club,
        date_from=date_from,
        date_to=date_to,
        contains=contains,
        limit=limit,
        api_key=api_backend,
    )


@app.get("/ffbridge/player-history")
def player_history(
    player_id: str = Query(..., pattern=r"^\d+$"),
    limit: int = Query(100, ge=1, le=500),
    score: str = Query("Scratch", pattern="^(Scratch|Handicap)$"),
    api_backend: str | None = Query(None),
) -> dict:
    """Return canonical score provenance and Results_URL for newest sessions."""
    return _run(
        reports.run_player_history,
        player_id=player_id,
        limit=limit,
        score=score,
        api_key=api_backend,
    )


@app.get("/ffbridge/board-results")
def board_results(
    session_id: str = Query(..., pattern=r"^\d+$"),
    board_number: int = Query(..., ge=1),
    force_refresh: bool = Query(False),
) -> dict:
    """Return all published results for one board across all session clubs."""
    return _run(
        boards.get_board_results,
        session_id=session_id,
        board_number=board_number,
        force_refresh=force_refresh,
    )


@app.get("/ffbridge/session-ranking")
def session_ranking(
    session_id: str = Query(..., pattern=r"^\d+$"),
    scope: str = Query("national", pattern="^(national|club)$"),
    club_code: str | None = Query(None),
    api_backend: str | None = Query(None),
) -> dict:
    """Official per-pair scratch and handicap ranking for one FFBridge session."""
    return _run(
        rankings.get_session_ranking,
        session_id=session_id,
        scope=scope,
        club_code=club_code,
        api_key=api_backend,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("FFBRIDGE_API_PORT", "8511"))
    started_at = datetime.now(timezone.utc)
    print(f"[ffbridge-api] start {started_at.isoformat()} port={port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
