"""Drive the ACBL Elo API with alternating Club / Tournament requests and log memory.

Run inside the API's container (Linux): it reads the server's RSS and anonymous
memory from /proc and the container total from the cgroup, after every request.

    python scripts/acbl_api_memory_drive.py --rounds 200 --out /tmp/drive.csv

Each round sends one request per variant in VARIANTS (Club and Tournament,
Players and Pairs, two date ranges), the same /acbl/report parameters the
Streamlit page sends. See ACBL_API_MEMORY.md for the results.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time
from datetime import datetime

import requests

VARIANTS = [
    {"club_or_tournament": "club", "rating_type": "Players", "date_range": "All time"},
    {"club_or_tournament": "tournament", "rating_type": "Players", "date_range": "All time"},
    {"club_or_tournament": "club", "rating_type": "Pairs", "date_range": "Last 1 year"},
    {"club_or_tournament": "tournament", "rating_type": "Pairs", "date_range": "Last 1 year"},
]

BASE_PARAMS = {
    "top_n": 100,
    "min_sessions": 10,
    "rating_method": "Latest",
    "moving_avg_days": 10,
    "elo_rating_type": "Current Rating (End of Session)",
    "online_filter": "All",
    "strata": "Open (unlimited)",
    "prior_sessions": 0,
    "platinum_events": False,
}


def server_pid() -> int:
    pids = []
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "uvicorn acbl_api_server" in cmdline and "nice" not in cmdline.split()[0]:
            pids.append(int(proc.name))
    if len(pids) != 1:
        raise RuntimeError(f"expected one uvicorn acbl_api_server process, found {pids}")
    return pids[0]


def proc_memory_gb(pid: int) -> dict[str, float]:
    fields = {}
    for line in pathlib.Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]:
        key, value = line.split(":", 1)
        fields[key.strip()] = int(value.split()[0]) * 1024 / 1024 ** 3
    return {
        "rss_gb": round(fields["Rss"], 3),
        "anon_gb": round(fields["Anonymous"], 3),
        "file_gb": round(fields["Rss"] - fields["Anonymous"], 3),
    }


def peak_rss_gb(pid: int, reset: bool = True) -> float:
    """VmHWM (peak RSS since the last reset); reset it so the next read is per request."""
    hwm = 0.0
    for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) / 1024 ** 2
    if reset:
        pathlib.Path(f"/proc/{pid}/clear_refs").write_text("5")
    return round(hwm, 3)


def cgroup_gb() -> float:
    return round(int(pathlib.Path("/sys/fs/cgroup/memory.current").read_text()) / 1024 ** 3, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8505")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--variants", default="0,1,2,3", help="indices into VARIANTS")
    args = parser.parse_args()
    variants = [VARIANTS[int(i)] for i in args.variants.split(",")]

    started = datetime.now()
    print(f"drive started {started.isoformat(timespec='seconds')}", flush=True)
    pid = server_pid()
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "round", "request", "club_or_tournament", "rating_type", "date_range",
            "status", "seconds", "server_seconds", "rss_gb", "anon_gb", "file_gb",
            "peak_rss_gb", "cgroup_gb", "frame_cache",
        ])
        writer.writerow([0, 0, "", "", "", "", "", "", *proc_memory_gb(pid).values(), peak_rss_gb(pid), cgroup_gb(), ""])
        request_no = 0
        for rnd in range(1, args.rounds + 1):
            for variant in variants:
                request_no += 1
                params = {**BASE_PARAMS, **variant}
                t0 = time.perf_counter()
                resp = requests.get(f"{args.base_url}/acbl/report", params=params, timeout=900)
                seconds = time.perf_counter() - t0
                server = {}
                server_seconds = ""
                if resp.status_code == 200:
                    payload = resp.json()
                    server = payload.get("server", {})
                    server_seconds = payload.get("perf", {}).get("total_seconds", "")
                mem = proc_memory_gb(pid)
                writer.writerow([
                    rnd, request_no, variant["club_or_tournament"], variant["rating_type"],
                    variant["date_range"], resp.status_code, round(seconds, 3), server_seconds,
                    mem["rss_gb"], mem["anon_gb"], mem["file_gb"], peak_rss_gb(pid), cgroup_gb(),
                    "|".join(sorted(str(k) for k in server.get("frame_cache", {}) or {})),
                ])
                fh.flush()
                if resp.status_code != 200:
                    print(f"request {request_no} {variant} -> {resp.status_code}: {resp.text[:300]}", flush=True)
            if rnd % 10 == 0 or rnd == 1:
                print(
                    f"round {rnd}: rss {mem['rss_gb']:.2f} GB anon {mem['anon_gb']:.2f} GB "
                    f"cgroup {cgroup_gb():.2f} GB last {seconds:.1f}s "
                    f"elapsed {(datetime.now() - started).total_seconds():.0f}s",
                    flush=True,
                )
    ended = datetime.now()
    print(
        f"drive ended {ended.isoformat(timespec='seconds')} elapsed {(ended - started).total_seconds():.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
