"""Upload the ACBL shrinkage sidecar JSON files to R2.

These are the small JSON sidecars produced by ``acbl_elo_ratings_create.py``
that ``acbl_api_server.py`` uses to compute Published (Bayesian-shrunk) Elo.
When parquets live in R2, this script pushes the matching sidecars to the same
prefix so the deployed API server can load them.

Reads R2 credentials from the same env vars used by ``acbl_api_server.py``:

    R2_BUCKET             (required)
    R2_ENDPOINT           (required, e.g. https://<acct>.r2.cloudflarestorage.com)
    R2_ACCESS_KEY_ID      (required)
    R2_SECRET_ACCESS_KEY  (required)
    R2_REGION             (optional, defaults to "auto")
    R2_PREFIX             (optional, defaults to "data")

Usage::

    python upload_shrinkage_sidecars_to_r2.py
    python upload_shrinkage_sidecars_to_r2.py --source path/to/data
    python upload_shrinkage_sidecars_to_r2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

SIDECAR_FILENAMES: tuple[str, ...] = (
    "acbl_tournament_elo_shrinkage.json",
    "acbl_club_elo_shrinkage.json",
)


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        sys.exit(f"ERROR: required env var {name} is not set")
    return val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "data",
        help="Local directory containing the sidecar JSON files (default: ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would be uploaded, without uploading",
    )
    args = parser.parse_args()

    started_at = datetime.now()
    print(f"[upload-shrinkage] start={started_at.isoformat()}")

    bucket = _require_env("R2_BUCKET")
    endpoint = _require_env("R2_ENDPOINT")
    access_key = _require_env("R2_ACCESS_KEY_ID")
    secret_key = _require_env("R2_SECRET_ACCESS_KEY")
    region = os.getenv("R2_REGION", "auto").strip() or "auto"
    prefix = os.getenv("R2_PREFIX", "data").strip().strip("/")

    source_dir: pathlib.Path = args.source
    if not source_dir.is_dir():
        sys.exit(f"ERROR: source directory does not exist: {source_dir}")

    targets: list[tuple[pathlib.Path, str]] = []
    for filename in SIDECAR_FILENAMES:
        path = source_dir / filename
        if not path.exists():
            print(f"  SKIP {filename} (not found in {source_dir})")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"ERROR: {path} is not valid JSON: {exc}")
        key = f"{prefix}/{filename}" if prefix else filename
        targets.append((path, key))
        size_kb = path.stat().st_size / 1024
        print(f"  READY {filename} ({size_kb:.1f} KB) -> s3://{bucket}/{key}")

    if not targets:
        sys.exit(f"ERROR: no sidecar files found under {source_dir}")

    if args.dry_run:
        print("[upload-shrinkage] dry-run; nothing uploaded.")
        return 0

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )

    uploaded = 0
    for path, key in targets:
        try:
            s3.upload_file(
                Filename=str(path),
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ContentType": "application/json"},
            )
        except (BotoCoreError, ClientError) as exc:
            sys.exit(f"ERROR: upload of {path.name} failed: {exc}")
        print(f"  UPLOADED s3://{bucket}/{key}")
        uploaded += 1

    ended_at = datetime.now()
    elapsed = (ended_at - started_at).total_seconds()
    print(
        f"[upload-shrinkage] uploaded={uploaded} elapsed={elapsed:.2f}s "
        f"end={ended_at.isoformat()}"
    )
    print(
        "\nNote: the ACBL API server caches sidecars in memory at module level,"
        " so restart the acbl-api container for the new values to take effect."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
