"""Nationwide FFBridge board-result retrieval backed by the Lancelot API."""
from __future__ import annotations

import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping


_ROOT = pathlib.Path(__file__).resolve().parent
_MLBRIDGE = next(
    (path for path in (_ROOT / "mlBridge", _ROOT.parent / "mlBridge") if path.is_dir()),
    None,
)
if _MLBRIDGE is None:
    raise FileNotFoundError("mlBridge not found at ./mlBridge or ../mlBridge")
if str(_MLBRIDGE.parent) not in sys.path:
    sys.path.insert(0, str(_MLBRIDGE.parent))

from mlBridge import mlBridgeFFLib  # noqa: E402


DATA_ROOT = pathlib.Path(
    os.environ.get("FFBRIDGE_CACHE_DIR", "data/ffbridge")
).resolve()
BOARD_RESULTS_CACHE_DIR = DATA_ROOT / "board_results_cache"


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_or_fetch(
    path: pathlib.Path,
    fetch: Any,
    *,
    force_refresh: bool,
) -> Any:
    if path.is_file() and not force_refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    value = fetch()
    _atomic_write_json(path, value)
    return value


def _team_id(row: Mapping[str, Any]) -> str | None:
    team = row.get("team")
    if not isinstance(team, Mapping) or team.get("id") is None:
        return None
    return str(team["id"])


def _orientation(row: Mapping[str, Any]) -> str:
    team = row.get("team")
    return str(
        row.get("orientation")
        or (team.get("orientation") if isinstance(team, Mapping) else "")
        or ""
    ).upper()


def _player(lineup: Mapping[str, Any], seat: str) -> dict[str, Any]:
    key = {
        "N": "northPlayer",
        "E": "eastPlayer",
        "S": "southPlayer",
        "W": "westPlayer",
    }[seat]
    raw = lineup.get(key)
    value = raw if isinstance(raw, Mapping) else {}
    name = " ".join(
        part.strip()
        for part in (str(value.get("firstName") or ""), str(value.get("lastName") or ""))
        if part.strip()
    )
    return {
        f"Player_ID_{seat}": (
            str(value["migrationId"])
            if value.get("migrationId") is not None
            else None
        ),
        f"Lancelot_Player_ID_{seat}": (
            str(value["id"]) if value.get("id") is not None else None
        ),
        f"Player_Name_{seat}": name or None,
    }


def _normalized_score(
    score: Mapping[str, Any],
    *,
    session_id: str,
    ranking_by_team: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    board = score.get("board")
    board_value = board if isinstance(board, Mapping) else {}
    lineup = score.get("lineup")
    lineup_value = lineup if isinstance(lineup, Mapping) else {}
    segment = lineup_value.get("segment")
    segment_value = segment if isinstance(segment, Mapping) else {}
    game = segment_value.get("game")
    game_value = game if isinstance(game, Mapping) else {}
    home = game_value.get("homeTeam")
    away = game_value.get("awayTeam")
    home_value = home if isinstance(home, Mapping) else {}
    away_value = away if isinstance(away, Mapping) else {}

    teams_by_orientation: dict[str, Mapping[str, Any]] = {}
    for team in (home_value, away_value):
        orientation = str(team.get("orientation") or "").upper()
        if orientation:
            teams_by_orientation[orientation] = team

    def team_field(orientation: str, field: str) -> Any:
        team = teams_by_orientation.get(orientation, {})
        team_id = str(team["id"]) if team.get("id") is not None else None
        ranking = ranking_by_team.get(team_id or "", {})
        if field == "id":
            return team_id
        if field == "label":
            return team.get("label") or (
                ranking.get("team", {}).get("label")
                if isinstance(ranking.get("team"), Mapping)
                else None
            )
        return ranking.get(field)

    row: dict[str, Any] = {
        "Session_ID": session_id,
        "Board": score.get("boardNumber") or board_value.get("boardNumber"),
        "Board_ID": (
            str(board_value["id"]) if board_value.get("id") is not None else None
        ),
        "Result_ID": str(score["id"]) if score.get("id") is not None else None,
        "PBN": board_value.get("deal"),
        "Contract": score.get("contract"),
        "Declarer": score.get("declarer"),
        "Result": score.get("result"),
        "Tricks": score.get("tricks"),
        "Lead": score.get("lead"),
        "Score_NS": score.get("nsScore"),
        "Score_EW": score.get("ewScore"),
        "Pct_NS": score.get("nsNote"),
        "Pct_EW": score.get("ewNote"),
        "Team_ID_NS": team_field("NS", "id"),
        "Team_ID_EW": team_field("EW", "id"),
        "Pair_NS": team_field("NS", "label"),
        "Pair_EW": team_field("EW", "label"),
        "Club_Code_NS": team_field("NS", "simultaneousId"),
        "Club_Code_EW": team_field("EW", "simultaneousId"),
        # Session-level Lancelot `rank` for this pair (national handicap
        # position on handicap series), copied onto every board row. Not a
        # board rank and not a scratch rank.
        "National_Rank_NS": team_field("NS", "rank"),
        "National_Rank_EW": team_field("EW", "rank"),
    }
    for seat in "NESW":
        row.update(_player(lineup_value, seat))
    return row


def get_board_results(
    session_id: str | int,
    board_number: int,
    *,
    force_refresh: bool = False,
    cache_dir: pathlib.Path | None = None,
    workers: int = 8,
    client: Any = mlBridgeFFLib,
) -> dict[str, Any]:
    """Return every published result for one board across all session clubs."""
    sid = str(session_id).strip()
    if not sid.isdigit():
        raise ValueError("session_id must contain only digits")
    if board_number < 1:
        raise ValueError("board_number must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    root = pathlib.Path(cache_dir or BOARD_RESULTS_CACHE_DIR) / sid
    ranking_path = root / "ranking.json"
    ranking = _load_or_fetch(
        ranking_path,
        lambda: client.get_session_ranking(int(sid)),
        force_refresh=force_refresh,
    )
    if not isinstance(ranking, list):
        raise ValueError(f"Lancelot ranking for session {sid} is not a list")
    ranking_rows = [row for row in ranking if isinstance(row, Mapping)]
    ranking_by_team = {
        team_id: row
        for row in ranking_rows
        if (team_id := _team_id(row)) is not None
    }
    covering_team_ids = sorted(
        team_id
        for team_id, row in ranking_by_team.items()
        if _orientation(row) == "NS"
    )
    if not covering_team_ids:
        raise ValueError(
            f"Session {sid} ranking has no NS teams to provide complete board coverage"
        )

    def load_team(team_id: str) -> list[Mapping[str, Any]]:
        path = root / "teams" / f"{team_id}.json"
        scores = _load_or_fetch(
            path,
            lambda: client.get_team_session_scores(int(team_id), int(sid)),
            force_refresh=force_refresh,
        )
        if not isinstance(scores, list):
            raise ValueError(
                f"Lancelot scores for session {sid}, team {team_id} are not a list"
            )
        return [score for score in scores if isinstance(score, Mapping)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        team_scores = executor.map(load_team, covering_team_ids)

    matching: dict[str, Mapping[str, Any]] = {}
    for scores in team_scores:
        for score in scores:
            raw_board = score.get("board")
            board = raw_board if isinstance(raw_board, Mapping) else {}
            number = score.get("boardNumber") or board.get("boardNumber")
            try:
                is_requested_board = int(number) == board_number
            except (TypeError, ValueError):
                is_requested_board = False
            if not is_requested_board:
                continue
            result_id = score.get("id")
            if result_id is None:
                raise ValueError(
                    f"Board result lacks id in session {sid}, board {board_number}"
                )
            matching[str(result_id)] = score

    rows = [
        _normalized_score(
            score,
            session_id=sid,
            ranking_by_team=ranking_by_team,
        )
        for score in matching.values()
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("Club_Code_NS") or ""),
            str(row.get("Team_ID_NS") or ""),
            str(row.get("Result_ID") or ""),
        )
    )
    return {
        "session_id": sid,
        "board_number": board_number,
        "rows": rows,
        "row_count": len(rows),
        "covering_team_count": len(covering_team_ids),
    }
