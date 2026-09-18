"""Solve PBNs with the original Bo Haglund dds-291 dds.dll. No ddss import."""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys

import polars as pl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_parquet", type=pathlib.Path)
    parser.add_argument("output_parquet", type=pathlib.Path)
    args = parser.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    mlbridge_parent = here.parent / "mlBridge"
    if mlbridge_parent.is_dir() and str(here.parent) not in sys.path:
        sys.path.insert(0, str(here.parent))
    from mlBridge import dds  # type: ignore

    pbns = pl.read_parquet(args.input_parquet)["PBN"].to_list()
    rows: list[dict[str, object]] = []
    max_tables = dds.MAXNOOFTABLES
    trump_filter = (ctypes.c_int * dds.DDS_STRAINS)(0, 0, 0, 0, 0)
    line = ctypes.create_string_buffer(80)
    dds.SetMaxThreads(0)
    for start in range(0, len(pbns), max_tables):
        chunk = pbns[start : start + max_tables]
        deals = dds.ddTableDealsPBN()
        table_res = dds.ddTablesRes()
        par_res = dds.allParResults()
        deals.noOfTables = len(chunk)
        for index, pbn in enumerate(chunk):
            deals.deals[index].cards = str(pbn).encode()
        status = dds.CalcAllTablesPBN(
            ctypes.pointer(deals),
            0,
            trump_filter,
            ctypes.pointer(table_res),
            ctypes.pointer(par_res),
        )
        if status != dds.RETURN_NO_FAULT:
            dds.ErrorMessage(status, line)
            raise RuntimeError(f"dds-291 CalcAllTablesPBN: {line.value.decode()}")
        for index, pbn in enumerate(chunk):
            table = table_res.results[index]
            rec: dict[str, object] = {"PBN": pbn}
            for seat_index, seat in enumerate("NESW"):
                for strain_index, strain in enumerate("SHDCN"):
                    rec[f"DD_{seat}_{strain}"] = int(
                        table.resTable[strain_index][seat_index]
                    )
            rows.append(rec)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.output_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
