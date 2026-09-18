"""Rebuild E:/W: Lancelot deals after the PbnToN 180-degree fix.

Re-converts raw board_deal strings (not cached N: PBNs), drops the old
wrong cache keys, fills DD/Par/SD for the new PBNs, and deletes Club
fragments for affected sessions so Stage 2 can re-augment them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl
from tqdm import tqdm

from ffbridge_quality_pipeline import (
    DEFAULT_SOURCE_DIR,
    EMBEDDED_DD_COLUMNS,
    _attach_sd_ev_from_unique_deals,
    _ddss_columns_from_table,
    _ddss_tables_for_pbns,
    _import_mlbridge,
    _read_json,
    audit_historical_cache,
    default_hrs_cache_path,
    load_hrs_cache,
    save_hrs_cache,
)

DD_BATCH = 500
SD_BATCH = 400
DEFAULT_CLUB_OUT = pathlib.Path(r"E:\bridge\data\ffbridge")
CLUB_FRAGMENT_DIRNAME = "club_session_fragments"


def legacy_pbn_to_n(bd: str) -> str:
    """The swapped E/W recipes that seated those deals 180 degrees off."""
    from endplay.types import Deal

    hands = bd[2:].split(" ")
    if len(hands) != 4:
        raise ValueError(f"PBN must have 4 hands: {bd}")
    prefix = bd[0]
    match prefix:
        case "N":
            pbn = bd
        case "E":
            pbn = "N:" + " ".join([hands[1], hands[2], hands[3], hands[0]])
        case "S":
            pbn = "N:" + " ".join([hands[2], hands[3], hands[0], hands[1]])
        case "W":
            pbn = "N:" + " ".join([hands[3], hands[0], hands[1], hands[2]])
        case _:
            raise ValueError(f"Invalid dealer: {prefix}")
    return Deal(pbn).to_pbn()


def _session_ew_rows(source_dir: pathlib.Path, session: Any) -> list[dict[str, Any]]:
    for team_id in session.expected_team_ids:
        path = (
            pathlib.Path(source_dir)
            / "results"
            / "teams"
            / str(team_id)
            / "session"
            / session.session_id
            / "scores.json"
        )
        if not path.is_file():
            continue
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        rows: list[dict[str, Any]] = []
        for score in payload:
            if not isinstance(score, dict):
                continue
            board = score.get("board")
            if not isinstance(board, dict):
                continue
            deal = str(board.get("deal") or "")
            if not deal or deal[:1] not in {"E", "W"}:
                continue
            try:
                board_number = int(score.get("boardNumber") or board.get("boardNumber"))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "session_id": session.session_id,
                    "Board": board_number,
                    "lancelot_deal": deal,
                }
            )
        if rows:
            return rows
    return []


def collect_ew_deals(
    source_dir: pathlib.Path, *, session_limit: int | None = None
) -> pl.DataFrame:
    ff_lib, _augment_lib = _import_mlbridge()
    report = audit_historical_cache(source_dir)
    sessions = [session for session in report.sessions if session.complete]
    if session_limit is not None:
        sessions = sessions[:session_limit]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(_session_ew_rows, source_dir, session): session
            for session in sessions
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Load E/W Lancelot deals"
        ):
            rows.extend(future.result())
    if not rows:
        raise SystemExit("No E:/W: Lancelot deals found")
    deals = pl.DataFrame(rows)
    new_pbns: list[str] = []
    old_pbns: list[str] = []
    dealers: list[str] = []
    vuls: list[str] = []
    for rec in deals.iter_rows(named=True):
        raw = rec["lancelot_deal"]
        new_pbns.append(ff_lib.PbnToN(raw))
        old_pbns.append(legacy_pbn_to_n(raw))
        dealers.append(ff_lib.BoardNumberToDealer(int(rec["Board"])))
        vuls.append(ff_lib.BoardNumberToVul(int(rec["Board"])))
    return deals.with_columns(
        pl.Series("PBN", new_pbns),
        pl.Series("old_PBN", old_pbns),
        pl.Series("Dealer", dealers),
        pl.Series("Vul", vuls),
    )


def _unique_new_deals(deals: pl.DataFrame) -> pl.DataFrame:
    return deals.select("PBN", "Dealer", "Vul", "Board").unique(
        subset=["PBN", "Dealer", "Vul"], maintain_order=True
    )


def _attach_ddss_tables(unique: pl.DataFrame) -> pl.DataFrame:
    pbns = unique["PBN"].unique(maintain_order=True).to_list()
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(pbns), DD_BATCH), desc="ddss new PBNs"):
        chunk = pbns[start : start + DD_BATCH]
        tables = _ddss_tables_for_pbns(chunk)
        for pbn, table in zip(chunk, tables):
            rows.append({"PBN": pbn, **_ddss_columns_from_table(table)})
    dd = pl.DataFrame(rows)
    return unique.join(dd, on="PBN", how="left")


def _drop_old_cache_keys(cache: pl.DataFrame, old_pbns: set[str], new_pbns: set[str]) -> pl.DataFrame:
    drop = old_pbns - new_pbns
    if not drop:
        return cache
    return cache.filter(~pl.col("PBN").is_in(list(drop)))


def _delete_session_fragments(root: pathlib.Path, session_ids: set[str]) -> int:
    deleted = 0
    for session_id in tqdm(sorted(session_ids), desc=f"Delete {root.name}"):
        path = root / session_id
        if path.is_dir():
            shutil.rmtree(path)
            deleted += 1
    return deleted


def repair_ew_rotation(
    source_dir: pathlib.Path,
    club_out: pathlib.Path,
    *,
    dry_run: bool = False,
    session_limit: int | None = None,
    sd_productions: int = 10,
) -> dict[str, int]:
    started = dt.datetime.now()
    print(
        f"[pbn-ew-repair] start {started.isoformat(timespec='seconds')}",
        flush=True,
    )
    deals = collect_ew_deals(source_dir, session_limit=session_limit)
    unique = _unique_new_deals(deals)
    old_pbns = {pbn for pbn in deals["old_PBN"].to_list() if pbn}
    new_pbns = {pbn for pbn in unique["PBN"].to_list() if pbn}
    session_ids = {str(item) for item in deals["session_id"].unique().to_list()}
    print(
        f"[pbn-ew-repair] {deals.height} E/W session-boards; "
        f"{unique.height} unique new PBNs; {len(session_ids)} sessions",
        flush=True,
    )
    cache_path = default_hrs_cache_path(source_dir)
    cache = load_hrs_cache(cache_path)
    if cache is None:
        raise SystemExit(f"hrs cache missing: {cache_path}")
    cache = _drop_old_cache_keys(cache, old_pbns, new_pbns)
    if not dry_run:
        save_hrs_cache(cache, cache_path)
    unique = _attach_ddss_tables(unique)
    if unique.select(
        pl.any_horizontal(
            [pl.col(column).is_null() for column in EMBEDDED_DD_COLUMNS]
        ).any()
    ).item():
        raise RuntimeError("ddss returned a null DD table")
    filled = 0
    for start in range(0, unique.height, SD_BATCH):
        chunk = unique.slice(start, SD_BATCH)
        chunk_started = time.time()
        _work, cache = _attach_sd_ev_from_unique_deals(
            chunk,
            hrs_cache_df=cache,
            cache_file_path=None if dry_run else cache_path,
            sd_productions=sd_productions,
            max_sd_adds=None,
        )
        filled += chunk.height
        elapsed = time.time() - chunk_started
        if elapsed > 30:
            print(
                f"[pbn-ew-repair] SD batch {start}-{start + chunk.height} "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )
    club_root = pathlib.Path(club_out) / CLUB_FRAGMENT_DIRNAME
    quality_root = pathlib.Path(source_dir).parent / "quality_cache" / "session_fragments"
    deleted_club = 0
    deleted_quality = 0
    if not dry_run:
        if club_root.is_dir():
            deleted_club = _delete_session_fragments(club_root, session_ids)
        if quality_root.is_dir():
            deleted_quality = _delete_session_fragments(quality_root, session_ids)
    stats = {
        "ew_boards": deals.height,
        "unique_new_pbns": unique.height,
        "sessions": len(session_ids),
        "old_keys_dropped": len(old_pbns - new_pbns),
        "sd_deals": filled,
        "club_fragments_deleted": deleted_club,
        "quality_fragments_deleted": deleted_quality,
    }
    ended = dt.datetime.now()
    print(
        f"[pbn-ew-repair] end {ended.isoformat(timespec='seconds')} "
        f"(elapsed {(ended - started).total_seconds():.1f}s) {stats}",
        flush=True,
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument(
        "--club-out",
        type=pathlib.Path,
        default=DEFAULT_CLUB_OUT,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session-limit", type=int, default=None)
    args = parser.parse_args()
    repair_ew_rotation(
        args.source_dir,
        args.club_out,
        dry_run=args.dry_run,
        session_limit=args.session_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
