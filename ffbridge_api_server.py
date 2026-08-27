"""First-party REST API for persisted FFBridge Elo reports."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query

import ffbridge_report_service as reports


app = FastAPI(title="FFBridge Elo API", version="1.0.0")


def _run(callable_, /, **kwargs):
    try:
        return callable_(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    info = _run(reports.dataset_info)
    return {
        "status": "ok",
        "service": "ffbridge-api",
        "dataset_built_at": info.get("built_at"),
        "result_rows": info.get("result_rows"),
        "score_provenance": info.get("score_provenance", {}),
        "results_links": info.get("results_links", {}),
        "quality_status": info.get("quality_status"),
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
    """Return Elo rows with deployed quality metrics and quality-cache status."""
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
    api_backend: str | None = Query(None),
) -> dict:
    """Return newest-first FFBridge sessions, including public Results_URL."""
    return _run(
        reports.run_player_history,
        player_id=player_id,
        limit=limit,
        api_key=api_backend,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("FFBRIDGE_API_PORT", "8511"))
    started_at = datetime.now(timezone.utc)
    print(f"[ffbridge-api] start {started_at.isoformat()} port={port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
