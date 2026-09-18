"""Export every Lancelot-vs-ddss DD mismatch from raw simultaneous sessions.

Reads Lancelot's original embedded DD tables (not the possibly corrected
hand-records cache). Solves the converted N: PBN with ddss, and also solves
the unrotated Lancelot deal string when it is a valid PBN. Writes one CSV
row per session-board that disagrees.

Do not run this while ffbridge_all / Stage 2 is using ddss. The ddss
DLL cannot be shared safely. dds-291 is solved in a subprocess.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl
from tqdm import tqdm

from ffbridge_quality_pipeline import (
    EMBEDDED_DD_COLUMNS,
    SEATS,
    _ddss_columns_from_table,
    _ddss_tables_for_pbns,
    _import_mlbridge,
    _par_rows_from_embedded_dd,
    _read_json,
    audit_historical_cache,
    default_hrs_cache_path,
    load_hrs_cache,
)

DD_BATCH = 1000
DEFAULT_OUTPUT = pathlib.Path(r"E:\bridge\data\ffbridge\lancelot_dd_mismatches.csv")


def _compact_dd(rec: dict[str, Any], prefix: str) -> str:
    """prefix is Lancelot_ or ddss_ in front of DD_N_S etc."""
    parts = []
    for seat in "NESW":
        cells = ",".join(
            f"{suit}={rec[f'{prefix}DD_{seat}_{suit}']}" for suit in "SHDCN"
        )
        parts.append(f"{seat}:{cells}")
    return " ".join(parts)


def _mismatch_cells(rec: dict[str, Any]) -> list[str]:
    cells = []
    for column in EMBEDDED_DD_COLUMNS:
        left = rec.get(f"Lancelot_{column}")
        right = rec.get(f"ddss_{column}")
        if left is None or right is None:
            continue
        if int(left) != int(right):
            cells.append(f"{column}:{int(left)}->{int(right)}")
    return cells


def _cards_key(pbn: str | None) -> str | None:
    if not pbn or ":" not in pbn:
        return None
    body = pbn.split(":", 1)[1]
    cards = "".join(ch for ch in body if ch not in {".", " "})
    if len(cards) != 52:
        return None
    return "".join(sorted(cards))


def _lancelot_dd_from_board(dds: Any) -> dict[str, Any] | None:
    if not isinstance(dds, dict):
        return None
    row: dict[str, Any] = {}
    for direction in SEATS:
        tricks = dds.get(direction)
        if not isinstance(tricks, dict):
            return None
        for raw_suit, value in tricks.items():
            suit = "N" if str(raw_suit).upper() == "NT" else str(raw_suit).upper()
            if suit in "SHDCN":
                row[f"DD_{direction}_{suit}"] = value
    if any(f"DD_{seat}_{suit}" not in row for seat in SEATS for suit in "SHDCN"):
        return None
    if "ParScore" in dds:
        row["ParScore"] = dds.get("ParScore")
    if "ParScore_NS" in dds:
        row["ParScore_NS"] = dds.get("ParScore_NS")
    return row


def _session_board_rows(source_dir: pathlib.Path, session: Any) -> list[dict[str, Any]]:
    by_board: dict[int, dict[str, Any]] = {}
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
        for score in payload:
            if not isinstance(score, dict):
                continue
            board = score.get("board")
            if not isinstance(board, dict):
                continue
            deal = board.get("deal")
            dd_row = _lancelot_dd_from_board(board.get("dds"))
            if not deal or dd_row is None:
                continue
            try:
                board_number = int(score.get("boardNumber") or board.get("boardNumber"))
            except (TypeError, ValueError):
                continue
            by_board[board_number] = {
                "Date": session.session_date,
                "session_id": session.session_id,
                "board_number": board_number,
                "Board": board_number,
                "lancelot_deal": str(deal),
                **dd_row,
            }
        if by_board:
            return list(by_board.values())
    return list(by_board.values())


def collect_lancelot_deals(
    source_dir: pathlib.Path, *, session_limit: int | None = None
) -> pl.DataFrame:
    ff_lib, _augment_lib = _import_mlbridge()
    BoardNumberToDealer = ff_lib.BoardNumberToDealer
    BoardNumberToVul = ff_lib.BoardNumberToVul
    PbnToN = ff_lib.PbnToN

    report = audit_historical_cache(source_dir)
    sessions = [session for session in report.sessions if session.complete]
    if session_limit is not None:
        sessions = sessions[:session_limit]
    rows: list[dict[str, Any]] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(_session_board_rows, source_dir, session): session
            for session in sessions
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Load Lancelot DD"
        ):
            try:
                session_rows = future.result()
            except Exception:
                skipped += 1
                continue
            if not session_rows:
                skipped += 1
                continue
            rows.extend(session_rows)
    print(
        f"[dd-mismatch] loaded {len(rows)} session-boards; skipped {skipped} sessions",
        flush=True,
    )
    if not rows:
        raise SystemExit("No Lancelot DD tables found")
    deals = pl.DataFrame(rows)
    converted: list[str | None] = []
    prefixes: list[str | None] = []
    dealers_deal: list[str | None] = []
    dealers_board: list[str | None] = []
    vuls: list[str | None] = []
    convert_ok: list[bool] = []
    same_cards: list[bool | None] = []
    for rec in deals.iter_rows(named=True):
        raw_deal = str(rec["lancelot_deal"])
        prefix = raw_deal[:1] if raw_deal else None
        prefixes.append(prefix)
        dealers_deal.append(prefix if prefix in {"N", "E", "S", "W"} else None)
        board = rec.get("Board")
        try:
            board_n = int(board)
            dealers_board.append(BoardNumberToDealer(board_n))
            vuls.append(BoardNumberToVul(board_n))
        except Exception:
            dealers_board.append(None)
            vuls.append(None)
        try:
            pbn = PbnToN(raw_deal)
            converted.append(pbn)
            convert_ok.append(True)
            same_cards.append(_cards_key(raw_deal) == _cards_key(pbn))
        except Exception:
            converted.append(None)
            convert_ok.append(False)
            same_cards.append(None)
    return deals.with_columns(
        pl.Series("PBN", converted),
        pl.Series("pbn_prefix", prefixes),
        pl.Series("Dealer", dealers_board),
        pl.Series("Dealer_from_deal", dealers_deal),
        pl.Series("Vul", vuls),
        pl.Series("pbn_convert_ok", convert_ok),
        pl.Series("same_cards_after_convert", same_cards),
    )


def _solve_unique_pbns(pbns: list[str]) -> pl.DataFrame:
    unique = [pbn for pbn in dict.fromkeys(pbns) if pbn]
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(unique), DD_BATCH), desc="ddss unique PBNs"):
        chunk = unique[start : start + DD_BATCH]
        tables = _ddss_tables_for_pbns(chunk)
        for pbn, table in zip(chunk, tables):
            rows.append({"PBN": pbn, **_ddss_columns_from_table(table)})
    return pl.DataFrame(rows)


def _solve_unique_pbns_dds291(pbns: list[str]) -> pl.DataFrame:
    """Solve in a subprocess so dds-291's dds.dll is not loaded beside ddss."""
    unique = [pbn for pbn in dict.fromkeys(pbns) if pbn]
    if not unique:
        return pl.DataFrame({"PBN": []})
    worker = pathlib.Path(__file__).resolve().with_name(
        "export_lancelot_dd_dds291_worker.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        incoming = pathlib.Path(tmp) / "pbns.parquet"
        outgoing = pathlib.Path(tmp) / "dds291.parquet"
        pl.DataFrame({"PBN": unique}).write_parquet(incoming)
        print(f"[dd-mismatch] dds-291 solving {len(unique)} unique PBNs", flush=True)
        subprocess.run(
            [sys.executable, str(worker), str(incoming), str(outgoing)],
            check=True,
        )
        return pl.read_parquet(outgoing)


def _rename_dd(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return frame.rename({column: f"{prefix}{column}" for column in EMBEDDED_DD_COLUMNS})


def _par_lookup(frame: pl.DataFrame, source_prefix: str, out_name: str) -> pl.DataFrame:
    _, augment_lib = _import_mlbridge()
    work = frame.rename(
        {f"{source_prefix}{column}": column for column in EMBEDDED_DD_COLUMNS}
    )
    par = _par_rows_from_embedded_dd(work, augment_lib).select(
        "PBN", "Dealer", "Vul", "ParScore"
    )
    return par.rename({"ParScore": out_name})


def build_mismatch_csv(
    source_dir: pathlib.Path,
    output: pathlib.Path,
    *,
    session_limit: int | None = None,
) -> pathlib.Path:
    started = dt.datetime.now()
    started_clock = time.perf_counter()
    print(f"[dd-mismatch] start {started.isoformat(timespec='seconds')}", flush=True)
    deals = collect_lancelot_deals(source_dir, session_limit=session_limit)
    print(f"[dd-mismatch] unique session-boards {deals.height}", flush=True)
    converted_pbns = [pbn for pbn in deals["PBN"].to_list() if pbn]
    ddss = _rename_dd(_solve_unique_pbns(converted_pbns), "ddss_")
    lancelot = _rename_dd(
        deals.select("lancelot_deal", *EMBEDDED_DD_COLUMNS).unique(
            subset=["lancelot_deal"], maintain_order=True
        ),
        "Lancelot_",
    )
    original_ok: list[str] = []
    for deal in lancelot["lancelot_deal"].to_list():
        if deal and deal[:1] in "NESW" and ":" in deal:
            original_ok.append(deal)
    original_ddss = pl.DataFrame(schema={"lancelot_deal": pl.String})
    if original_ok:
        solved = _solve_unique_pbns(original_ok).rename({"PBN": "lancelot_deal"})
        original_ddss = _rename_dd(solved, "ddss_original_")
    dds291_raw = _solve_unique_pbns_dds291([*converted_pbns, *original_ok])
    dds291 = _rename_dd(
        dds291_raw.filter(pl.col("PBN").is_in(converted_pbns)), "dds291_"
    )
    original_dds291 = pl.DataFrame(schema={"lancelot_deal": pl.String})
    if original_ok:
        original_dds291 = _rename_dd(
            dds291_raw.filter(pl.col("PBN").is_in(original_ok)).rename(
                {"PBN": "lancelot_deal"}
            ),
            "dds291_original_",
        )
    joined = (
        deals.join(lancelot, on="lancelot_deal", how="left")
        .join(ddss, on="PBN", how="left")
        .join(dds291, on="PBN", how="left")
        .join(original_ddss, on="lancelot_deal", how="left")
        .join(original_dds291, on="lancelot_deal", how="left")
    )
    mismatch_flags = pl.any_horizontal(
        [
            pl.col(f"Lancelot_{column}").cast(pl.Int16, strict=False)
            != pl.col(f"ddss_{column}").cast(pl.Int16, strict=False)
            for column in EMBEDDED_DD_COLUMNS
        ]
    )
    ddss_vs_291 = pl.all_horizontal(
        [
            pl.col(f"ddss_{column}").cast(pl.Int16, strict=False)
            == pl.col(f"dds291_{column}").cast(pl.Int16, strict=False)
            for column in EMBEDDED_DD_COLUMNS
        ]
    )
    lancelot_vs_291 = pl.all_horizontal(
        [
            pl.col(f"Lancelot_{column}").cast(pl.Int16, strict=False)
            == pl.col(f"dds291_{column}").cast(pl.Int16, strict=False)
            for column in EMBEDDED_DD_COLUMNS
        ]
    )
    original_match = pl.all_horizontal(
        [
            pl.col(f"Lancelot_{column}").cast(pl.Int16, strict=False)
            == pl.col(f"ddss_original_{column}").cast(pl.Int16, strict=False)
            for column in EMBEDDED_DD_COLUMNS
            if f"ddss_original_{column}" in joined.columns
        ]
    ) if original_ddss.height else pl.lit(None)
    joined = joined.with_columns(
        mismatch_flags.alias("dd_mismatch"),
        original_match.alias("lancelot_matches_ddss_on_original_deal"),
        ddss_vs_291.alias("ddss_matches_dds291"),
        lancelot_vs_291.alias("lancelot_matches_dds291"),
        (pl.col("Dealer") != pl.col("Dealer_from_deal")).alias("dealer_mismatch"),
    )
    mismatches = joined.filter(pl.col("dd_mismatch"))
    print(f"[dd-mismatch] mismatched session-boards {mismatches.height}", flush=True)
    if mismatches.is_empty():
        output.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame().write_csv(output)
        return output
    par_src = mismatches.filter(
        pl.col("PBN").is_not_null()
        & pl.col("Dealer").is_not_null()
        & pl.col("Vul").is_not_null()
    )
    if par_src.is_empty():
        lancelot_par = pl.DataFrame(
            schema={
                "PBN": pl.String,
                "Dealer": pl.String,
                "Vul": pl.String,
                "ParScore_Lancelot": pl.Int32,
            }
        )
        ddss_par = pl.DataFrame(
            schema={
                "PBN": pl.String,
                "Dealer": pl.String,
                "Vul": pl.String,
                "ParScore_ddss": pl.Int32,
            }
        )
    else:
        lancelot_par = _par_lookup(
            par_src.select(
                "PBN",
                "Dealer",
                "Vul",
                *[f"Lancelot_{column}" for column in EMBEDDED_DD_COLUMNS],
            ).unique(subset=["PBN", "Dealer", "Vul"], maintain_order=True),
            "Lancelot_",
            "ParScore_Lancelot",
        )
        ddss_par = _par_lookup(
            par_src.select(
                "PBN",
                "Dealer",
                "Vul",
                *[f"ddss_{column}" for column in EMBEDDED_DD_COLUMNS],
            ).unique(subset=["PBN", "Dealer", "Vul"], maintain_order=True),
            "ddss_",
            "ParScore_ddss",
        )
    mismatches = mismatches.join(
        lancelot_par, on=["PBN", "Dealer", "Vul"], how="left"
    ).join(ddss_par, on=["PBN", "Dealer", "Vul"], how="left")
    cache_path = default_hrs_cache_path(source_dir)
    cache = load_hrs_cache(cache_path)
    if cache is not None and not cache.is_empty() and "DD_N_C" in cache.columns:
        cache_dd = _rename_dd(
            cache.select("PBN", *EMBEDDED_DD_COLUMNS).unique(
                subset=["PBN"], maintain_order=True
            ),
            "cache_",
        )
        mismatches = mismatches.join(cache_dd, on="PBN", how="left")
        cache_is_ddss = pl.all_horizontal(
            [
                pl.col(f"cache_{column}").cast(pl.Int16, strict=False)
                == pl.col(f"ddss_{column}").cast(pl.Int16, strict=False)
                for column in EMBEDDED_DD_COLUMNS
            ]
        )
        mismatches = mismatches.with_columns(cache_is_ddss.alias("corrected_in_cache"))
    else:
        mismatches = mismatches.with_columns(pl.lit(None).alias("corrected_in_cache"))
    compact_lancelot: list[str] = []
    compact_ddss: list[str] = []
    compact_291: list[str] = []
    cells: list[str] = []
    counts: list[int] = []
    for rec in mismatches.iter_rows(named=True):
        named = {
            f"Lancelot_{column}": rec[f"Lancelot_{column}"]
            for column in EMBEDDED_DD_COLUMNS
        }
        named.update(
            {f"ddss_{column}": rec[f"ddss_{column}"] for column in EMBEDDED_DD_COLUMNS}
        )
        miss = _mismatch_cells(named)
        compact_lancelot.append(_compact_dd(rec, "Lancelot_"))
        compact_ddss.append(_compact_dd(rec, "ddss_"))
        compact_291.append(_compact_dd(rec, "dds291_"))
        cells.append(";".join(miss))
        counts.append(len(miss))
    mismatches = mismatches.with_columns(
        pl.Series("Lancelot_DD", compact_lancelot),
        pl.Series("ddss_DD", compact_ddss),
        pl.Series("dds291_DD", compact_291),
        pl.Series("mismatch_cells", cells),
        pl.Series("mismatch_count", counts),
        (
            pl.col("ParScore_Lancelot").cast(pl.Int32, strict=False)
            != pl.col("ParScore_ddss").cast(pl.Int32, strict=False)
        ).alias("par_mismatch"),
    )
    published = "ParScore" if "ParScore" in mismatches.columns else None
    out_cols = [
        "Date",
        "session_id",
        "board_number",
        "Board",
        "lancelot_deal",
        "PBN",
        "pbn_prefix",
        "pbn_convert_ok",
        "same_cards_after_convert",
        "Dealer",
        "Dealer_from_deal",
        "dealer_mismatch",
        "Vul",
        "Lancelot_DD",
        "ddss_DD",
        "dds291_DD",
        "mismatch_cells",
        "mismatch_count",
        "lancelot_matches_ddss_on_original_deal",
        "ddss_matches_dds291",
        "lancelot_matches_dds291",
        "corrected_in_cache",
        "ParScore_Lancelot",
        "ParScore_ddss",
        "par_mismatch",
    ]
    if published:
        out_cols.insert(-3, "Lancelot_published_ParScore")
        mismatches = mismatches.rename({published: "Lancelot_published_ParScore"})
    if "ParScore_NS" in mismatches.columns:
        out_cols.insert(-3, "Lancelot_published_ParScore_NS")
        mismatches = mismatches.rename(
            {"ParScore_NS": "Lancelot_published_ParScore_NS"}
        )
    out_cols.extend(
        [f"Lancelot_{column}" for column in EMBEDDED_DD_COLUMNS]
        + [f"ddss_{column}" for column in EMBEDDED_DD_COLUMNS]
        + [f"dds291_{column}" for column in EMBEDDED_DD_COLUMNS]
    )
    present = [column for column in out_cols if column in mismatches.columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    mismatches.select(present).sort(
        ["Date", "session_id", "board_number"]
    ).write_csv(output)
    elapsed = time.perf_counter() - started_clock
    print(
        f"[dd-mismatch] wrote {output} rows={mismatches.height} "
        f"end {dt.datetime.now().isoformat(timespec='seconds')} elapsed {elapsed:.1f}s",
        flush=True,
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=pathlib.Path(r"E:\bridge\data\ffbridge\data"),
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_mismatch_csv(
        args.source_dir, args.output, session_limit=args.session_limit
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
