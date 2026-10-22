# Elo_Ratings

Unofficial bridge game Elo ratings. Python project (no build step, no test suite, no linter configured).

## Cursor Cloud specific instructions

### Services

This repo contains three independently-runnable services (see the `railway-*.toml` files for the canonical Railway start commands):

| Service | File | Dev run command | Port used in setup |
|---|---|---|---|
| FFBridge Elo Streamlit app | `streamlit_app_ffbridge_elo_ratings.py` | `streamlit run streamlit_app_ffbridge_elo_ratings.py --server.port 8501 --server.address 0.0.0.0 --server.headless true` | 8501 |
| ACBL Elo API (FastAPI) | `acbl_api_server.py` | `uvicorn acbl_api_server:app --host 0.0.0.0 --port 8000` | 8000 |
| ACBL Elo Streamlit app | `streamlit_app_acbl_elo_ratings.py` | `ACBL_API_BASE_URL=http://localhost:8000 streamlit run streamlit_app_acbl_elo_ratings.py --server.port 8502 --server.address 0.0.0.0 --server.headless true` | 8502 |

The update script creates a virtualenv at `.venv`; activate it with `. .venv/bin/activate` before running any service.

### Non-obvious caveats

- **FFBridge app is fully self-contained.** It fetches live data from the public FFBridge Lancelot API (`https://api-lancelot.ffbridge.fr`) and needs no token for the default "All Tournaments" browse. `FFBRIDGE_BEARER_TOKEN` is only needed for the optional Classic API source. This is the service to use for a no-secrets end-to-end demo.
- **FFBridge runtime cache lives in `./data/ffbridge/`** (override with `FFBRIDGE_CACHE_DIR`). This directory is created/populated at runtime and is NOT tracked in git — do not commit it.
- **ACBL app runs in "API-only mode".** `streamlit_app_acbl_elo_ratings.py` requires `ACBL_API_BASE_URL` to point at a running `acbl_api_server.py`; it has no local data path of its own.
- **The ACBL API server needs parquet data that is not in the repo.** `/health` always works, but `/acbl/report` and `/acbl/detail` read `data/acbl_{club,tournament}_elo_ratings.parquet`. Provide them either by setting the `R2_*` env vars (Cloudflare R2, see `.env.example`) or by placing the parquet files in `./data/`. Without data these endpoints return HTTP 500 (`Missing file: .../data/acbl_club_elo_ratings.parquet`), and the ACBL Streamlit app surfaces that as an "ACBL API request failed ... 500" error — expected when no data source is configured.
- **Do not upgrade Streamlit past the pin.** `streamlit==1.53.1` is pinned because `streamlit >= 1.56.0` breaks `streamlit-aggrid` (tables render blank). See the comment in `requirements.txt`.
- **No automated tests or linter** are configured. A reasonable build/sanity check is `python -m py_compile <source>.py` on the changed files.
