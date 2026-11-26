"""Brill Bridge Challenge Elo dataset: fetch, score, rate, rank.

Brill (https://brill.aalborgdata.dk) runs daily eight-board events. Every
entrant, human or robot, sits South at their own table beside three Brill
robots, and all entrants play the same boards. There are no partnerships to
rate, only entrants.

Data path
  brill_client (postmortem-brill)   signed-in reads of the site's database
  fetch_new_tournaments()           closed events not built yet -> board rows
  augment_boards()                  double dummy, par, lead cost, matchpoints, cross-IMPs
  boards/{tournament_id}.parquet    one fragment per event (the cache)
  brill_boards.parquet              all fragments, one row per entrant per board
  brill_sessions.parquet            one row per entrant per event, with Elo

Elo is replayed over every event in time order: an entrant's score is the mean
of their board matchpoint percentages against the other tables (for IMP events
as well), and the expectation comes from their rating against the mean rating
of the rest of the field. A view of one event series or one scoring replays
Elo over those events alone (rated_selection), and every view is shown on one
fixed scale taken from the all-event ratings (elo_scale).

Env:
  BRILL_ELO_CACHE_DIR   dataset directory (default data/brill next to this file)
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys
import threading
import urllib.parse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from elo_common import (
    DEFAULT_ELO,
    K_FACTOR,
    calculate_elo_from_percentage,
    get_elo_title,
    zscore_to_chess,
)

_APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("BRILL_ELO_CACHE_DIR", str(_APP_DIR / "data" / "brill")))
FRAGMENT_DIR = CACHE_DIR / "boards"
BOARDS_PATH = CACHE_DIR / "brill_boards.parquet"
SESSIONS_PATH = CACHE_DIR / "brill_sessions.parquet"
METADATA_PATH = CACHE_DIR / "brill_elo_metadata.json"

# Bump when a fragment's columns or their meaning change; old fragments are rebuilt.
SCHEMA_VERSION = 1

SEATS = ("N", "E", "S", "W")
STRAINS = "SHDCN"
DD_COLUMNS = tuple(f"DD_{seat}_{strain}" for seat in SEATS for strain in STRAINS)
HUMAN, ROBOT = "Human", "Robot"
# Every entrant sits South: they play the hand when North or South declares
# and make the opening lead when East declares.
ENTRANT_DECLARES = ("N", "S")
ENTRANT_LEADS_AGAINST = "E"
POSTMORTEM_URL = "https://brill.postmortem.chat"

_VUL_TO_MLBRIDGE = {"None": "None", "NS": "N_S", "EW": "E_W", "All": "Both"}
_CONTRACT_RE = re.compile(r"^([1-7])(NT|N|[SHDC])(XX|X)?$")
_CARD_RE = re.compile(r"^[SHDC][AKQJT2-9]$")
_EVENT_DATE_RE = re.compile(r"\s*[—–-]\s*\d{4}-\d{2}-\d{2}\s*$")
_RESULT_COLUMNS = (
    "board_number,contract,declarer,tricks,ns_score,dealer,vulnerability,deal_pbn,play_pbn,"
    "participant_id,participant:challenge_participants!participant_id("
    "id,display_name,participant_type,robot_name,user_id,join_order)"
)
_STANDING_COLUMNS = "tournament_id,display_name,user_id,final_rank"
_ID_CHUNK = 40
# Differences up to each bound score 0, 1, 2, ... IMPs.
_IMP_BOUNDS = np.array(
    [10, 40, 80, 120, 160, 210, 260, 310, 360, 420, 490, 590, 740, 890, 1090, 1290,
     1490, 1740, 1990, 2240, 2490, 2990, 3490, 3990]
)

_build_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Imports that live outside this repo
# ---------------------------------------------------------------------------

def _sibling(name: str, marker: str) -> pathlib.Path:
    """A project folder beside (dev tree) or inside (container) this one."""
    for root in (_APP_DIR / name, _APP_DIR.parent / name):
        if (root / marker).exists():
            return root
    raise FileNotFoundError(f"{name} not found at {_APP_DIR / name} or {_APP_DIR.parent / name}")


def _brill_modules() -> Tuple[Any, Any]:
    """(brill_client, brill_pbn) from postmortem-brill, which holds the sign-in."""
    root = str(_sibling("postmortem-brill", "brill_client.py"))
    if root not in sys.path:
        sys.path.append(root)
    import brill_client
    import brill_pbn

    return brill_client, brill_pbn


def _augment_lib() -> Any:
    root = str(_sibling("mlBridge", "mlBridgeAugmentLib.py").parent)
    if root not in sys.path:
        sys.path.append(root)
    from mlBridge import mlBridgeAugmentLib

    return mlBridgeAugmentLib


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def player_identity(participant: Dict[str, Any]) -> Tuple[str, str, str]:
    """(Player_ID, Player_Name, Player_Type) of one challenge participant.

    A human keeps one id across events and name changes: a short hash of the
    site's account id, so the account id itself is never published. A robot is
    its robot name.
    """
    name = (participant.get("display_name") or "").strip()
    if participant.get("participant_type") == "robot":
        robot = (participant.get("robot_name") or name or "Robot").strip()
        return f"R:{robot}", name or robot, ROBOT
    user_id = participant.get("user_id")
    if user_id:
        return "H:" + hashlib.sha256(str(user_id).encode()).hexdigest()[:10], name or "Unknown", HUMAN
    return f"N:{name or 'Unknown'}", name or "Unknown", HUMAN


def event_type(name: Optional[str]) -> str:
    """'Daily MP — 2026-11-13' -> 'Daily MP'."""
    return _EVENT_DATE_RE.sub("", (name or "").strip()) or "Unknown"


def _most_common(values: Iterable[Any]) -> Any:
    counted = Counter(v for v in values if v not in (None, ""))
    return counted.most_common(1)[0][0] if counted else None


def tournament_board_rows(
    tournament: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    standings: Sequence[Dict[str, Any]],
    brill_pbn: Any,
) -> List[Dict[str, Any]]:
    """One row per entrant per board, before double-dummy work."""
    from endplay.types import Deal

    rank_by_user = {s["user_id"]: s.get("final_rank") for s in standings if s.get("user_id")}
    rank_by_name = {s.get("display_name"): s.get("final_rank") for s in standings if not s.get("user_id")}
    event_time = tournament.get("starts_at") or tournament.get("created_at") or ""
    boards_in_event = max((r.get("board_number") or 0 for r in results), default=0)

    # A robot's row often has no deal and a placeholder dealer and vulnerability;
    # the rows that carry the deal say what the board is.
    board_facts: Dict[int, Dict[str, Any]] = {}
    for number in {r.get("board_number") for r in results}:
        with_deal = [r for r in results if r.get("board_number") == number and r.get("deal_pbn")]
        deal = _most_common(r["deal_pbn"] for r in with_deal)
        if not deal:
            continue
        same = [r for r in with_deal if r["deal_pbn"] == deal]
        dealer = _most_common(r.get("dealer") for r in same)
        if not re.match(r"^[NESW]:", deal):
            deal = f"{dealer}:{deal}"
        vul = _most_common(brill_pbn.vulnerable_to_pbn(r.get("vulnerability")) for r in same)
        board_facts[number] = {
            "PBN": Deal(deal).to_pbn(),
            "Dealer": dealer,
            "Vul": _VUL_TO_MLBRIDGE[vul],
        }

    rows: List[Dict[str, Any]] = []
    for result in results:
        facts = board_facts.get(result.get("board_number"))
        if facts is None or result.get("ns_score") is None:
            continue
        participant = result.get("participant") or {}
        player_id, player_name, player_type = player_identity(participant)
        contract = brill_pbn.contract_to_pbn(result.get("contract")).upper().replace(" ", "")
        parsed = _CONTRACT_RE.match(contract)
        declarer = result.get("declarer") if result.get("declarer") in SEATS else None
        if parsed and declarer:
            level, strain, dbl = int(parsed.group(1)), parsed.group(2)[0], parsed.group(3) or ""
            tricks = int(result.get("tricks") or 0)
            cards = [str(c).upper() for c in brill_pbn._tokens(result.get("play_pbn"))]
            lead = cards[0] if cards and _CARD_RE.match(cards[0]) else None
            contract_text = f"{level}{strain}{dbl}"
        else:  # passed out
            level = strain = declarer = tricks = lead = None
            dbl, contract_text = "", "Pass"
        official_rank = (
            rank_by_user.get(participant.get("user_id")) if participant.get("user_id")
            else rank_by_name.get(participant.get("display_name"))
        )
        rows.append({
            "tournament_id": tournament["id"],
            "Event": tournament.get("name") or "",
            "Event_Type": event_type(tournament.get("name")),
            "Scoring": "IMP" if tournament.get("scoring_method") == "imp" else "MP",
            "Event_Time": event_time,
            "Date": event_time[:10],
            "Boards_In_Event": boards_in_event,
            "Board": result.get("board_number"),
            "Player_ID": player_id,
            "Player_Name": player_name,
            "Player_Type": player_type,
            "Table": participant.get("join_order"),
            "Site_Rank": official_rank,
            **facts,
            "Contract": contract_text,
            "BidLvl": level,
            "BidSuit": strain,
            "Dbl": dbl,
            "Declarer_Direction": declarer,
            "Tricks": tricks,
            "Score_NS": int(result["ns_score"]),
            "Lead": lead,
        })
    return rows


_ROW_SCHEMA = {
    "tournament_id": pl.String, "Event": pl.String, "Event_Type": pl.String, "Scoring": pl.String,
    "Event_Time": pl.String, "Date": pl.String, "Boards_In_Event": pl.Int32, "Board": pl.Int32,
    "Player_ID": pl.String, "Player_Name": pl.String, "Player_Type": pl.String, "Table": pl.Int32,
    "Site_Rank": pl.Int32, "PBN": pl.String, "Dealer": pl.String, "Vul": pl.String,
    "Contract": pl.String, "BidLvl": pl.Int32, "BidSuit": pl.String, "Dbl": pl.String,
    "Declarer_Direction": pl.String, "Tricks": pl.Int32, "Score_NS": pl.Int32, "Lead": pl.String,
}


def list_closed_tournaments() -> List[Dict[str, Any]]:
    """Every closed field event on the site, oldest first."""
    client, _ = _brill_modules()
    # brill_client has no public "all tournaments" read; its paginated select is the same call.
    rows = client._select("tournaments", {
        "select": client._TOURNAMENT_COLUMNS,
        "tournament_type": f'in.({",".join(client.FIELD_TYPES)})',
        "order": "created_at.asc",
    })
    return [row for row in rows if client.is_closed(row)]


def built_tournament_ids() -> set[str]:
    if not FRAGMENT_DIR.is_dir():
        return set()
    return {path.stem for path in FRAGMENT_DIR.glob("*.parquet")}


def _skipped_ids() -> Dict[str, str]:
    """Closed events with nothing to rate (no challenge, or no readable boards): id -> reason."""
    return dict(read_metadata().get("skipped") or {})


def fetch_board_rows(
    tournaments: Sequence[Dict[str, Any]],
    progress: Optional[Callable[[Iterable], Iterable]] = None,
) -> Tuple[pl.DataFrame, Dict[str, str]]:
    """Board rows for these tournaments, and the ones that had none: id -> reason."""
    client, brill_pbn = _brill_modules()
    ids = [t["id"] for t in tournaments]
    challenge_of: Dict[str, str] = {}
    standings: Dict[str, List[Dict[str, Any]]] = {}
    for start in range(0, len(ids), _ID_CHUNK):
        chunk = f'in.({",".join(ids[start:start + _ID_CHUNK])})'
        for row in client._select("tournament_rounds", {
            "select": "tournament_id,round_number,challenge_id",
            "tournament_id": chunk,
            "challenge_id": "not.is.null",
            "order": "tournament_id,round_number",
        }):
            challenge_of.setdefault(row["tournament_id"], row["challenge_id"])
        for row in client._select("tournament_players", {"select": _STANDING_COLUMNS, "tournament_id": chunk}):
            standings.setdefault(row["tournament_id"], []).append(row)

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    for tournament in (progress(tournaments) if progress else tournaments):
        challenge_id = challenge_of.get(tournament["id"])
        if not challenge_id:
            skipped[tournament["id"]] = "no field round"
            continue
        results = client._select("challenge_board_results", {
            "select": _RESULT_COLUMNS,
            "challenge_id": f"eq.{challenge_id}",
            "order": "board_number.asc",
        })
        found = tournament_board_rows(tournament, results, standings.get(tournament["id"], []), brill_pbn)
        if found:
            rows.extend(found)
        else:
            skipped[tournament["id"]] = "no board results"
    return pl.DataFrame(rows, schema=_ROW_SCHEMA), skipped


# ---------------------------------------------------------------------------
# Augment: double dummy, par, lead cost, matchpoints, cross-IMPs
# ---------------------------------------------------------------------------

def _dd_tables(pbns: Sequence[str], augment_lib: Any) -> pl.DataFrame:
    from endplay.types import Deal

    tables = augment_lib.solve_dd_for_deals([Deal(pbn) for pbn in pbns])
    records = []
    for pbn, table in zip(pbns, tables):
        by_seat = table.to_list(player_major=True)
        records.append({"PBN": pbn, **{
            f"DD_{seat}_{strain}": int(by_seat[i][j])
            for i, seat in enumerate(SEATS) for j, strain in enumerate(STRAINS)
        }})
    return pl.DataFrame(records, schema={"PBN": pl.String, **{c: pl.Int32 for c in DD_COLUMNS}})


def _par_rows(deals: pl.DataFrame, augment_lib: Any) -> pl.DataFrame:
    """Par score (NS view), par strains and a readable par contract list per deal."""
    records = []
    for rec in deals.iter_rows(named=True):
        table = augment_lib._list_to_ddtable(
            [[int(rec[f"DD_{seat}_{strain}"]) for strain in STRAINS] for seat in SEATS])
        parlist = augment_lib.par(
            table, augment_lib.VulToEndplayVul_d[rec["Vul"]], augment_lib.DealerToEndPlayDealer_d[rec["Dealer"]])
        strains, texts = set(), []
        for contract in parlist:
            strain = STRAINS[int(contract.denom)]
            strains.add(strain)
            result = "=" if contract.result == 0 else f"{contract.result:+d}"
            text = f"{contract.level}{strain}{contract.penalty.abbr}{result} {contract.declarer.abbr}"
            if text not in texts:
                texts.append(text)
        records.append({
            "PBN": rec["PBN"], "Dealer": rec["Dealer"], "Vul": rec["Vul"],
            "ParScore_NS": int(parlist.score),
            "_Par_Strains": sorted(strains),
            "ParContracts": ", ".join(texts) or "Pass",
        })
    return pl.DataFrame(records, schema={
        "PBN": pl.String, "Dealer": pl.String, "Vul": pl.String, "ParScore_NS": pl.Int32,
        "_Par_Strains": pl.List(pl.String), "ParContracts": pl.String,
    })


def _dd_tricks_expr() -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Int32)
    for seat in SEATS:
        for strain in STRAINS:
            expr = (
                pl.when((pl.col("Declarer_Direction") == seat) & (pl.col("BidSuit") == strain))
                .then(pl.col(f"DD_{seat}_{strain}"))
                .otherwise(expr)
            )
    return expr


def cross_imps(scores: np.ndarray) -> np.ndarray:
    """Each table's mean IMP result against every other table of the board."""
    if len(scores) < 2:
        return np.full(len(scores), np.nan)
    diff = scores[:, None] - scores[None, :]
    imps = np.sign(diff) * np.searchsorted(_IMP_BOUNDS, np.abs(diff), side="right")
    return imps.sum(axis=1) / (len(scores) - 1)


def add_field_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Pct_NS (0-100 matchpoint percentage) and IMP_NS (cross-IMPs) against the
    other tables that played the same board of the same event."""
    board = ["tournament_id", "Board"]
    tables = pl.len().over(board)
    df = df.with_columns(
        pl.when(tables > 1)
        .then((pl.col("Score_NS").rank(method="average").over(board) - 1) / (tables - 1) * 100)
        .otherwise(None)
        .alias("Pct_NS")
    )
    imps = (
        df.group_by(board, maintain_order=True)
        .agg(pl.col("Player_ID"), pl.col("Score_NS").alias("_scores"))
        .with_columns(
            pl.col("_scores").map_elements(
                lambda s: cross_imps(np.asarray(s, dtype=np.int64)).tolist(),
                return_dtype=pl.List(pl.Float64),
            ).alias("IMP_NS")
        )
        .drop("_scores")
        .explode("Player_ID", "IMP_NS")
    )
    return df.join(imps, on=[*board, "Player_ID"], how="left", validate="1:1")


def augment_boards(rows: pl.DataFrame) -> pl.DataFrame:
    """Add the double-dummy, par, lead and field-comparison columns to board rows."""
    if rows.is_empty():
        return rows
    augment_lib = _augment_lib()
    scores_d = augment_lib.precompute_contract_score_tables()[1]

    df = rows.join(_dd_tables(rows["PBN"].unique(maintain_order=True).to_list(), augment_lib), on="PBN", how="left")
    deals = df.select("PBN", "Dealer", "Vul", *DD_COLUMNS).unique(subset=["PBN", "Dealer", "Vul"], maintain_order=True)
    df = df.join(_par_rows(deals, augment_lib), on=["PBN", "Dealer", "Vul"], how="left")

    declarer_ns = pl.col("Declarer_Direction").is_in(["N", "S"])
    passed_out = pl.col("Declarer_Direction").is_null()
    vul_declarer = (
        pl.when(declarer_ns).then(pl.col("Vul").is_in(["N_S", "Both"]))
        .otherwise(pl.col("Vul").is_in(["E_W", "Both"]))
    )
    df = df.with_columns(
        _dd_tricks_expr().alias("DD_Tricks"),
        (pl.col("Tricks") - pl.col("BidLvl") - 6).alias("Result"),
        vul_declarer.alias("_Vul_Declarer"),
    ).with_columns(
        (pl.col("Tricks") - pl.col("DD_Tricks")).alias("DD_Tricks_Diff"),
        # Score of the contract played double dummy. As in mlBridge, a contract
        # that goes down is scored doubled, so a sacrifice equals the par score.
        pl.struct("BidLvl", "BidSuit", "DD_Tricks", "_Vul_Declarer").map_elements(
            lambda r: scores_d.get((r["BidLvl"], r["BidSuit"], r["DD_Tricks"], r["_Vul_Declarer"])),
            return_dtype=pl.Int32,
        ).alias("_DD_Score_Declarer"),
    ).with_columns(
        pl.when(passed_out).then(0)
        .when(declarer_ns).then(pl.col("_DD_Score_Declarer"))
        .otherwise(-pl.col("_DD_Score_Declarer"))
        .cast(pl.Int32).alias("DD_Score_NS"),
        pl.when(declarer_ns).then(pl.col("ParScore_NS")).otherwise(-pl.col("ParScore_NS")).alias("_Par_Declarer"),
    ).with_columns(
        pl.when(passed_out).then(None)
        .otherwise(pl.col("_Par_Strains").list.contains(pl.col("BidSuit")))
        .alias("Is_Par_Suit"),
        (pl.col("DD_Score_NS") >= pl.col("ParScore_NS")).alias("Is_Par_Contract"),
        (pl.col("Score_NS") >= pl.col("ParScore_NS")).alias("Is_Par_Score"),
        (pl.col("Score_NS") - pl.col("ParScore_NS")).alias("Par_Diff_NS"),
        pl.when(passed_out).then(None).otherwise(pl.col("_Par_Declarer") < 0).alias("Sacrifice_Opportunity"),
        pl.when(passed_out).then(None)
        .otherwise((pl.col("_Par_Declarer") < 0) & (pl.col("_Par_Declarer") == pl.col("_DD_Score_Declarer")))
        .alias("Is_Sacrifice"),
    )
    df = augment_lib.add_dd_lead_cost(df)
    df = add_field_scores(df)
    return df.drop(*DD_COLUMNS, "_Par_Strains", "_Vul_Declarer", "_DD_Score_Declarer", "_Par_Declarer")


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

def session_table(boards: pl.DataFrame) -> pl.DataFrame:
    """One row per entrant per event: boards played, score against the field, ranks."""
    sessions = boards.group_by("tournament_id", "Player_ID", maintain_order=True).agg(
        pl.col("Event").first(), pl.col("Event_Type").first(), pl.col("Scoring").first(),
        pl.col("Event_Time").first(), pl.col("Date").first(),
        pl.col("Boards_In_Event").first(),
        pl.col("Player_Name").first(), pl.col("Player_Type").first(),
        pl.col("Site_Rank").first(),
        pl.len().cast(pl.Int32).alias("Boards"),
        pl.col("Pct_NS").mean().alias("Pct"),
        pl.col("IMP_NS").mean().alias("IMP"),
        pl.col("Score_NS").sum().alias("Score"),
    )
    event = pl.col("tournament_id")
    return sessions.with_columns(
        pl.len().over(event).cast(pl.Int32).alias("Field_Size"),
        pl.col("Pct").rank(method="min", descending=True).over(event).cast(pl.Int32).alias("Pct_Rank"),
        pl.col("IMP").rank(method="min", descending=True).over(event).cast(pl.Int32).alias("IMP_Rank"),
    ).sort("Event_Time", "tournament_id", "Pct_Rank", "Player_ID")


def rate_sessions(sessions: pl.DataFrame, k_factor: float = K_FACTOR) -> pl.DataFrame:
    """Replay Elo over the events in time order.

    Each event moves a rating by K x (share of the boards played) x (score -
    expectation). The score is the entrant's mean board percentage; the
    expectation is the standard Elo curve of their rating against the mean
    rating of the other entrants.
    """
    ratings: Dict[str, float] = {}
    before: List[Optional[float]] = []
    after: List[Optional[float]] = []
    field: List[Optional[float]] = []
    for event in sessions.partition_by("tournament_id", maintain_order=True):
        ids = event["Player_ID"].to_list()
        pcts = event["Pct"].to_list()
        weights = (event["Boards"] / event["Boards_In_Event"]).to_list()
        start = [ratings.get(player, DEFAULT_ELO) for player in ids]
        total = sum(start)
        for player, pct, weight, rating in zip(ids, pcts, weights, start):
            before.append(rating)
            if pct is None or len(ids) < 2:
                after.append(rating)
                field.append(None)
                continue
            others = (total - rating) / (len(ids) - 1)
            ratings[player] = calculate_elo_from_percentage(rating, pct, others, k_factor=k_factor * weight)
            after.append(ratings[player])
            field.append(others)
    return sessions.with_columns(
        pl.Series("Elo_Before", before, dtype=pl.Float64),
        pl.Series("Elo_After", after, dtype=pl.Float64),
        pl.Series("Field_Elo", field, dtype=pl.Float64),
    ).with_columns((pl.col("Elo_After") - pl.col("Elo_Before")).alias("Elo_Delta"))


# ---------------------------------------------------------------------------
# Build and load
# ---------------------------------------------------------------------------

def read_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.is_file():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _write_atomic(frame: pl.DataFrame, path: pathlib.Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temp)
    temp.replace(path)


def _fragment_is_current(path: pathlib.Path) -> bool:
    return pl.read_parquet_schema(path).get("Schema_Version") is not None and (
        pl.scan_parquet(path).select(pl.col("Schema_Version").first()).collect().item() == SCHEMA_VERSION
    )


def update_dataset(
    progress: Optional[Callable[[Iterable], Iterable]] = None,
    limit: Optional[int] = None,
    rebuild: bool = False,
    max_age_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the fragments of closed events that have none, then the two dataset files.

    Returns the metadata written beside them. One build runs at a time. With
    max_age_hours, a dataset checked against the site more recently than that
    is returned as it is. When the site has nothing new, only the metadata's
    checked_at moves, so built_at still says when the data last changed.
    """
    with _build_lock:
        if max_age_hours is not None and not rebuild and dataset_exists():
            age = dataset_age_hours()
            if age is not None and age < max_age_hours:
                return read_metadata()
        FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
        outdated = [
            path for path in FRAGMENT_DIR.glob("*.parquet")
            if rebuild or not _fragment_is_current(path)
        ]
        for path in outdated:
            path.unlink()
        closed = list_closed_tournaments()
        skipped = {} if rebuild else _skipped_ids()
        have = built_tournament_ids()
        wanted = [t for t in closed if t["id"] not in have and t["id"] not in skipped]
        if limit is not None:
            wanted = wanted[:limit]
        added = 0
        if wanted:
            rows, newly_skipped = fetch_board_rows(wanted, progress)
            skipped.update(newly_skipped)
            augmented = augment_boards(rows).with_columns(pl.lit(SCHEMA_VERSION, dtype=pl.Int32).alias("Schema_Version"))
            for fragment in augmented.partition_by("tournament_id", maintain_order=True):
                _write_atomic(fragment, FRAGMENT_DIR / f"{fragment['tournament_id'][0]}.parquet")
                added += 1

        checked_at = datetime.now(timezone.utc).isoformat()
        if not added and not outdated and dataset_exists():
            metadata = {**read_metadata(), "checked_at": checked_at, "closed_on_site": len(closed), "skipped": skipped}
            METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metadata

        fragments = sorted(FRAGMENT_DIR.glob("*.parquet"))
        if not fragments:
            raise RuntimeError("Brill returned no closed events with board results.")
        boards = pl.concat([pl.read_parquet(path) for path in fragments], how="vertical").sort(
            "Event_Time", "tournament_id", "Board", "Table", "Player_ID")
        sessions = rate_sessions(session_table(boards))
        _write_atomic(boards, BOARDS_PATH)
        _write_atomic(sessions, SESSIONS_PATH)
        metadata = {
            "built_at": checked_at,
            "checked_at": checked_at,
            "schema_version": SCHEMA_VERSION,
            "tournaments": len(fragments),
            "tournaments_added": added,
            "closed_on_site": len(closed),
            "board_rows": boards.height,
            "session_rows": sessions.height,
            "first_date": boards["Date"].min(),
            "last_date": boards["Date"].max(),
            "skipped": skipped,
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata


def dataset_exists() -> bool:
    return BOARDS_PATH.is_file() and SESSIONS_PATH.is_file() and METADATA_PATH.is_file()


def load_dataset() -> Dict[str, Any]:
    """{boards, sessions, metadata} as last built."""
    if not dataset_exists():
        raise FileNotFoundError(f"No Brill Elo dataset in {CACHE_DIR}. Run build_brill_elo_parquets.py.")
    return {
        "boards": pl.read_parquet(BOARDS_PATH),
        "sessions": pl.read_parquet(SESSIONS_PATH),
        "metadata": read_metadata(),
    }


def dataset_age_hours() -> Optional[float]:
    """Hours since the dataset was last checked against the site."""
    checked_at = read_metadata().get("checked_at")
    if not checked_at:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(checked_at)).total_seconds() / 3600


# ---------------------------------------------------------------------------
# Filters and the leaderboard
# ---------------------------------------------------------------------------

# Players with at least this many events (over all events) set the Elo scale.
SCALE_MIN_GAMES = 10
DATE_RANGE_OPTIONS = ("All time", "Last 7 days", "Last 30 days", "Last 90 days")
PLAYER_TYPE_OPTIONS = ("All", "Humans", "Robots")
ALL_EVENTS = "All Events"

# Ranking name -> (rank column, columns that lead the table after the identity columns).
RANKINGS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "Elo": ("Elo_Rank", ()),
    "Declarer": ("Declarer_Rank", ("Declarer_Rank", "DD_Tricks_Diff_Avg", "DD_Tricks_Declared", "Declarer_Pct")),
    "Defender": ("Defender_Rank", ("Defender_Rank", "Defense_Tricks_Diff_Avg", "Boards_Defended", "Defender_Pct")),
    "Opening Lead": ("Leader_Rank", ("Leader_Rank", "Lead_Cost_Avg", "Lead_Best_Pct", "Leads_Scored")),
    "Par": ("Par_Rank", (
        "Par_Rank", "Par_Contract_Rate_Pct", "Par_Suit_Rate_Pct", "Sacrifice_Rate_Pct",
        "Par_Score_Rate_Pct", "Par_Diff_Avg")),
}
LEADERBOARD_COLUMNS = (
    "Elo_Rank", "Player_Name", "Type", "Elo", "Title", "Games", "Boards", "Avg_Pct", "Avg_IMP", "Wins",
    "Declarer_Rank", "DD_Tricks_Diff_Avg", "DD_Tricks_Declared", "Declarer_Pct",
    "Defender_Rank", "Defense_Tricks_Diff_Avg", "Boards_Defended", "Defender_Pct",
    "Leader_Rank", "Lead_Cost_Avg", "Lead_Best_Pct", "Leads_Scored",
    "Par_Rank", "Par_Contract_Rate_Pct", "Par_Suit_Rate_Pct", "Sacrifice_Rate_Pct",
    "Par_Score_Rate_Pct", "Par_Diff_Avg",
    "Last_Played", "Elo_Raw", "Player_ID",
)
METRIC_DEFINITIONS = {
    "Elo": "Rating replayed over the selected events alone, as it stood after the player's latest "
           "event in the date range, on a fixed chess-style scale: 1500 and 400 are the mean and "
           "standard deviation of the all-event ratings of the players with 10 or more events.",
    "Avg_Pct": "Mean board matchpoint percentage against the other tables, IMP events included.",
    "Avg_IMP": "Mean cross-IMPs per board against the other tables.",
    "Wins": "Events where the site ranked the player first.",
    "DD_Tricks_Diff_Avg": "Declarer play: mean of tricks taken minus double-dummy tricks on boards "
                          "the player's side declared (the entrant plays North's contracts too). Higher is better.",
    "Declarer_Pct": "Mean board percentage on boards the player's side declared.",
    "Defense_Tricks_Diff_Avg": "Defence: mean of double-dummy tricks minus the tricks declarer took, "
                               "on boards East or West declared. Higher is better. The partner is a robot.",
    "Defender_Pct": "Mean board percentage on boards East or West declared.",
    "Lead_Cost_Avg": "Opening lead: mean tricks given up against the best double-dummy lead, on "
                     "boards East declared (the entrant leads). 0 is best, lower is better.",
    "Lead_Best_Pct": "Share of opening leads that were a best double-dummy lead.",
    "Par_Contract_Rate_Pct": "Share of boards where the final contract, played double dummy, scores at least par for the player.",
    "Par_Suit_Rate_Pct": "Share of the player's side's contracts that are in a par strain.",
    "Sacrifice_Rate_Pct": "Of the contracts declared when par was a sacrifice for the player's side, "
                          "the share that are the par sacrifice.",
    "Par_Score_Rate_Pct": "Share of boards where the actual score was at least par.",
    "Par_Diff_Avg": "Mean of actual score minus par score.",
}


def date_range_start(choice: str, today: Optional[date] = None) -> Optional[str]:
    if choice == "All time":
        return None
    days = int(re.search(r"\d+", choice).group())
    return ((today or datetime.now(timezone.utc).date()) - timedelta(days=days)).isoformat()


def filter_frame(
    frame: pl.DataFrame,
    event_type_choice: str = ALL_EVENTS,
    scoring: str = "All",
    date_from: Optional[str] = None,
) -> pl.DataFrame:
    """Boards or sessions of the selected events."""
    if event_type_choice != ALL_EVENTS:
        frame = frame.filter(pl.col("Event_Type") == event_type_choice)
    if scoring != "All":
        frame = frame.filter(pl.col("Scoring") == scoring)
    if date_from:
        frame = frame.filter(pl.col("Date") >= date_from)
    return frame


def rated_selection(
    boards: pl.DataFrame,
    event_type_choice: str = ALL_EVENTS,
    scoring: str = "All",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """(boards, sessions) of one event series and scoring, with Elo replayed
    over those events alone.

    An MP-only or IMP-only rating is its own rating: it never sees the other
    events. Filtering the all-event sessions instead would show the all-event
    rating as it stood after the player's last selected event.
    """
    selected = filter_frame(boards, event_type_choice, scoring)
    return selected, rate_sessions(session_table(selected))


def elo_scale(sessions: pl.DataFrame, min_games: int = SCALE_MIN_GAMES) -> Tuple[float, float]:
    """(mean, standard deviation) of the latest ratings of the players with at
    least min_games events in sessions.

    Taken once from the all-event replay, this is the fixed reference every
    view is scaled against, so a displayed Elo means the same whichever
    filters and Minimum Games are chosen.
    """
    latest = (
        sessions.sort("Event_Time", "tournament_id").group_by("Player_ID")
        .agg(pl.len().alias("Games"), pl.col("Elo_After").last())
        .filter(pl.col("Games") >= min_games)["Elo_After"]
    )
    if latest.len() < 2:
        raise ValueError(f"Fewer than two players have {min_games} or more events; no Elo scale.")
    return float(latest.mean()), float(latest.std(ddof=0))


def _rank(column: str, alias: str, descending: bool = True) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_not_null())
        .then(pl.col(column).rank(method="min", descending=descending))
        .otherwise(None)
        .cast(pl.Int32)
        .alias(alias)
    )


def _pct(expr: pl.Expr) -> pl.Expr:
    return (expr.cast(pl.Float64).mean() * 100).round(1)


def leaderboard(
    boards: pl.DataFrame,
    sessions: pl.DataFrame,
    scale: Tuple[float, float],
    min_games: int = 1,
) -> pl.DataFrame:
    """One row per player over the given (already selected) boards and sessions.

    scale is the (mean, standard deviation) of raw ratings that maps to 1500
    and 400 on the displayed Elo (see elo_scale). Players with fewer than
    min_games events are left out before anything is ranked, so every rank is
    among the listed players; the Elo does not depend on who is listed.
    """
    if sessions.is_empty():
        return pl.DataFrame(schema={column: pl.String for column in LEADERBOARD_COLUMNS})
    declares = pl.col("Declarer_Direction").is_in(list(ENTRANT_DECLARES))
    defends = pl.col("Declarer_Direction").is_in(["E", "W"])
    leads = pl.col("Declarer_Direction") == ENTRANT_LEADS_AGAINST

    played = sessions.sort("Event_Time", "tournament_id").group_by("Player_ID").agg(
        pl.col("Player_Name").last(),
        pl.col("Player_Type").last().alias("Type"),
        pl.len().cast(pl.Int32).alias("Games"),
        pl.col("Boards").sum().cast(pl.Int32),
        (pl.col("Site_Rank") == 1).sum().cast(pl.Int32).alias("Wins"),
        pl.col("Date").last().alias("Last_Played"),
        pl.col("Elo_After").last().alias("Elo_Raw"),
    ).filter(pl.col("Games") >= min_games)
    if played.is_empty():
        return pl.DataFrame(schema={column: pl.String for column in LEADERBOARD_COLUMNS})

    quality = boards.group_by("Player_ID").agg(
        pl.col("Pct_NS").mean().round(2).alias("Avg_Pct"),
        pl.col("IMP_NS").mean().round(2).alias("Avg_IMP"),
        pl.col("DD_Tricks_Diff").filter(declares).cast(pl.Float64).mean().round(3).alias("DD_Tricks_Diff_Avg"),
        pl.col("DD_Tricks_Diff").filter(declares).count().cast(pl.Int32).alias("DD_Tricks_Declared"),
        pl.col("Pct_NS").filter(declares).mean().round(2).alias("Declarer_Pct"),
        (-pl.col("DD_Tricks_Diff").filter(defends).cast(pl.Float64)).mean().round(3).alias("Defense_Tricks_Diff_Avg"),
        pl.col("DD_Tricks_Diff").filter(defends).count().cast(pl.Int32).alias("Boards_Defended"),
        pl.col("Pct_NS").filter(defends).mean().round(2).alias("Defender_Pct"),
        pl.col("DD_Lead_Cost").filter(leads).cast(pl.Float64).mean().round(3).alias("Lead_Cost_Avg"),
        _pct(pl.col("Lead_Is_DD_Best").filter(leads)).alias("Lead_Best_Pct"),
        pl.col("DD_Lead_Cost").filter(leads).count().cast(pl.Int32).alias("Leads_Scored"),
        _pct(pl.col("Is_Par_Contract")).alias("Par_Contract_Rate_Pct"),
        _pct(pl.col("Is_Par_Suit").filter(declares)).alias("Par_Suit_Rate_Pct"),
        _pct(pl.col("Is_Sacrifice").filter(declares & pl.col("Sacrifice_Opportunity"))).alias("Sacrifice_Rate_Pct"),
        _pct(pl.col("Is_Par_Score")).alias("Par_Score_Rate_Pct"),
        pl.col("Par_Diff_NS").cast(pl.Float64).mean().round(1).alias("Par_Diff_Avg"),
    )
    table = played.join(quality, on="Player_ID", how="left")

    mean, sd = scale
    table = table.with_columns(
        pl.col("Elo_Raw").map_elements(lambda r: int(round(zscore_to_chess(r, mean, sd))), return_dtype=pl.Int32).alias("Elo"),
    ).with_columns(
        pl.col("Elo").map_elements(get_elo_title, return_dtype=pl.String).alias("Title"),
        pl.col("Elo_Raw").round(1),
        _rank("Elo_Raw", "Elo_Rank"),
        _rank("DD_Tricks_Diff_Avg", "Declarer_Rank"),
        _rank("Defense_Tricks_Diff_Avg", "Defender_Rank"),
        _rank("Lead_Cost_Avg", "Leader_Rank", descending=False),
        _rank("Par_Contract_Rate_Pct", "Par_Rank"),
    )
    return table.select(LEADERBOARD_COLUMNS).sort("Elo_Rank", "Player_Name")


def order_for_ranking(table: pl.DataFrame, ranking: str) -> pl.DataFrame:
    """The leaderboard sorted for one ranking, its own columns moved up front."""
    rank_column, lead_columns = RANKINGS[ranking]
    identity = ["Player_Name", "Type"]
    front = [rank_column, *identity, *[c for c in lead_columns if c != rank_column]]
    rest = [c for c in table.columns if c not in front]
    return table.select(*front, *rest).sort(rank_column, "Elo_Rank", nulls_last=True)


def filter_players(table: pl.DataFrame, player_type: str = "All", name: str = "") -> pl.DataFrame:
    """Display filters. Ranks and Elo stay as computed over all listed players."""
    if player_type == "Humans":
        table = table.filter(pl.col("Type") == HUMAN)
    elif player_type == "Robots":
        table = table.filter(pl.col("Type") == ROBOT)
    needle = (name or "").strip().casefold()
    if needle:
        table = table.filter(pl.col("Player_Name").str.to_lowercase().str.contains(needle.lower(), literal=True))
    return table


def player_sessions(sessions: pl.DataFrame, player_id: str) -> pl.DataFrame:
    """One player's events, newest first."""
    return (
        sessions.filter(pl.col("Player_ID") == player_id)
        .sort("Event_Time", "tournament_id", descending=True)
        .select(
            "Date", "Event", "Scoring", "Boards",
            pl.col("Pct").round(2), pl.col("IMP").round(2),
            "Site_Rank", "Pct_Rank", "Field_Size",
            pl.col("Field_Elo").round(1),
            pl.col("Elo_Before").round(1), pl.col("Elo_After").round(1), pl.col("Elo_Delta").round(1),
            pl.format(
                "{}/?tournament={}&entrant={}", pl.lit(POSTMORTEM_URL), pl.col("tournament_id"),
                pl.col("Player_Name").map_elements(urllib.parse.quote, return_dtype=pl.String),
            ).alias("Postmortem_URL"),
            pl.col("tournament_id").alias("_tournament_id"),
        )
    )


def event_standings(sessions: pl.DataFrame, tournament_id: str) -> pl.DataFrame:
    """Every entrant of one event, best percentage first."""
    return (
        sessions.filter(pl.col("tournament_id") == tournament_id)
        .sort("Pct_Rank", "Player_Name")
        .select(
            "Pct_Rank", "Site_Rank", "Player_Name", pl.col("Player_Type").alias("Type"), "Boards",
            pl.col("Pct").round(2), pl.col("IMP").round(2), "Score",
            pl.col("Elo_Before").round(1), pl.col("Elo_After").round(1), pl.col("Elo_Delta").round(1),
        )
    )


def player_event_boards(boards: pl.DataFrame, tournament_id: str, player_id: str) -> pl.DataFrame:
    """One player's boards of one event."""
    return (
        boards.filter((pl.col("tournament_id") == tournament_id) & (pl.col("Player_ID") == player_id))
        .sort("Board")
        .select(
            "Board", "Dealer", "Vul", "Contract", pl.col("Declarer_Direction").alias("Declarer"),
            "Tricks", "DD_Tricks", "DD_Tricks_Diff", "Lead", "DD_Lead_Cost",
            "Score_NS", "ParScore_NS", "ParContracts",
            pl.col("Pct_NS").round(1), pl.col("IMP_NS").round(2), "PBN",
        )
    )
