"""Experimental replay of the ACBL player Elo with switchable changes.

A copy of ``mlBridgeAugmentLib._elo_player_kernel`` (same field-relative
expectation, provisional boost, section scale, stratum and field-strength
dampening, MP-limit K scale, clamp) with these options added:

- ``merged``: one rating per player instead of one per direction. The
  production kernel keeps separate N/S and E/W ratings for each player.
- ``kmul``: an extra per-row K multiplier (e.g. by masterpoint colour).
- ``init``: a per-player starting rating instead of 1500 for everyone.
- ``k_base``, ``boost``, ``boost_until``, ``elo_scale``.

With the defaults it reproduces the production player ratings (up to the
global-stdev seeds, which production reads from the previous run's file).
Outputs are pre-session ratings and pre-session session counts per seat.
"""
from __future__ import annotations

import pathlib
import sys

import numba
import numpy as np
import polars as pl

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import mlBridge.mlBridgeAugmentLib as lib  # noqa: E402

lib._elo_ensure_jit()
_pairwise = lib._elo_pairwise_sum_f32
_stratum = lib._elo_stratum_scale_k
_strength = lib._elo_field_strength_scale_k


@numba.njit(cache=False, nogil=True)
def _field(buf, n, tmp, pop_mean, gsd, anchors):
    total = np.float64(_pairwise(buf, 0, n))
    if n >= 2:
        mean = total / n
        mean32 = np.float32(mean)
        for k in range(n):
            d = buf[k] - mean32
            tmp[k] = d * d
        stdev = np.float64(np.sqrt(_pairwise(tmp, 0, n) / np.float32(n)))
        st = _stratum(stdev / gsd, anchors[0], anchors[1], anchors[2])
        z = (mean - pop_mean) / gsd
        fs = _strength(z, anchors[3], anchors[4], anchors[5])
    else:
        st = 1.0
        fs = 1.0
    return total, st, fs


@numba.njit(cache=False, nogil=True)
def _kernel(proc_order, sess_starts, sess_ends, board, idx_n, idx_s, idx_e, idx_w,
            pct, scale, kmul, init, anc_n, anc_s, anc_e, anc_w, alpha, null_slots, max_len,
            pop_mean, k_base, boost, boost_until,
            elo_scale, min_r, max_r, gsd_ns, gsd_ew, anchors,
            r_n, r_s, r_e, r_w, c_n, c_s, c_e, c_w, x_n, x_s, x_e, x_w):
    n_slots = init.shape[0]
    ratings = init.copy()
    bp = np.zeros(n_slots, dtype=np.int32)
    nsess = np.zeros(n_slots, dtype=np.int32)
    delta = np.zeros(n_slots, dtype=np.float64)
    bc = np.zeros(n_slots, dtype=np.int32)
    mark = np.full(n_slots, -1, dtype=np.int64)
    part = np.zeros(n_slots, dtype=np.int64)
    buf_ns = np.zeros(max_len, dtype=np.float32)
    buf_ew = np.zeros(max_len, dtype=np.float32)
    tmp = np.zeros(max_len, dtype=np.float32)
    two = np.float32(2.0)
    for s in range(sess_starts.shape[0]):
        a = sess_starts[s]
        b = sess_ends[s]
        npart = 0
        for q in range(a, b):
            i = proc_order[q]
            for seat in range(4):
                if seat == 0:
                    idx = idx_n[i]
                    av = anc_n[i]
                elif seat == 1:
                    idx = idx_s[i]
                    av = anc_s[i]
                elif seat == 2:
                    idx = idx_e[i]
                    av = anc_e[i]
                else:
                    idx = idx_w[i]
                    av = anc_w[i]
                if mark[idx] != s:
                    mark[idx] = s
                    part[npart] = idx
                    npart += 1
                    delta[idx] = 0.0
                    bc[idx] = 0
                    # Mean reversion toward an informed anchor, once per session.
                    if alpha > 0.0 and np.isfinite(av):
                        ratings[idx] = ratings[idx] + np.float32(alpha * (av - ratings[idx]))
            r_n[i] = ratings[idx_n[i]]
            r_s[i] = ratings[idx_s[i]]
            r_e[i] = ratings[idx_e[i]]
            r_w[i] = ratings[idx_w[i]]
            c_n[i] = nsess[idx_n[i]]
            c_s[i] = nsess[idx_s[i]]
            c_e[i] = nsess[idx_e[i]]
            c_w[i] = nsess[idx_w[i]]
        bs = a
        while bs < b:
            be = bs + 1
            while be < b and board[proc_order[be]] == board[proc_order[be - 1]]:
                be += 1
            nb = be - bs
            for off in range(nb):
                i = proc_order[bs + off]
                buf_ns[off] = (ratings[idx_n[i]] + ratings[idx_s[i]]) / two
                buf_ew[off] = (ratings[idx_e[i]] + ratings[idx_w[i]]) / two
            sum_ns, st_ns, fs_ns = _field(buf_ns, nb, tmp, pop_mean, gsd_ns, anchors)
            sum_ew, st_ew, fs_ew = _field(buf_ew, nb, tmp, pop_mean, gsd_ew, anchors)
            for off in range(nb):
                i = proc_order[bs + off]
                ns_self = np.float64(buf_ns[off])
                ew_self = np.float64(buf_ew[off])
                if nb >= 2:
                    fa_ns = (sum_ns - ns_self) / (nb - 1)
                    fa_ew = (sum_ew - ew_self) / (nb - 1)
                else:
                    fa_ns = ns_self
                    fa_ew = ew_self
                e_ns = 1.0 / (1.0 + 10.0 ** (-(ns_self - fa_ns) / elo_scale))
                e_ew = 1.0 / (1.0 + 10.0 ** (-(ew_self - fa_ew) / elo_scale))
                sv = np.float64(pct[i])
                if np.isnan(sv) or sv < 0.0 or sv > 1.0:
                    continue
                sc = np.float64(scale[i]) * np.float64(kmul[i])
                res_ns = sv - e_ns
                res_ew = (1.0 - sv) - e_ew
                pn = idx_n[i]
                ps = idx_s[i]
                pe = idx_e[i]
                pw = idx_w[i]
                k_n = k_base * (boost if bp[pn] < boost_until else 1.0) * sc * st_ns * fs_ns
                k_s = k_base * (boost if bp[ps] < boost_until else 1.0) * sc * st_ns * fs_ns
                k_e = k_base * (boost if bp[pe] < boost_until else 1.0) * sc * st_ew * fs_ew
                k_w = k_base * (boost if bp[pw] < boost_until else 1.0) * sc * st_ew * fs_ew
                delta[pn] += 0.5 * k_n * res_ns
                delta[ps] += 0.5 * k_s * res_ns
                delta[pe] += 0.5 * k_e * res_ew
                delta[pw] += 0.5 * k_w * res_ew
                bc[pn] += 1
                bc[ps] += 1
                bc[pe] += 1
                bc[pw] += 1
            bs = be
        for p in range(npart):
            idx = part[p]
            nr = ratings[idx] + np.float32(delta[idx])
            if nr < min_r:
                ratings[idx] = min_r
            elif nr > max_r:
                ratings[idx] = max_r
            else:
                ratings[idx] = nr
            bp[idx] += bc[idx]
            if bc[idx] > 0:
                nsess[idx] += 1
        for k in range(null_slots.shape[0]):
            z = null_slots[k]
            ratings[z] = init[z]
            bp[z] = 0
            nsess[z] = 0
        for q in range(a, b):
            i = proc_order[q]
            x_n[i] = ratings[idx_n[i]]
            x_s[i] = ratings[idx_s[i]]
            x_e[i] = ratings[idx_e[i]]
            x_w[i] = ratings[idx_w[i]]


class Layout:
    """Arrays that do not depend on the options; build once per frame."""

    def __init__(self, df: pl.DataFrame):
        self.n = df.height
        (self.proc_order, self.starts, self.ends, _codes, _use_seen, _nc, self.board) = lib._elo_session_layout(df)
        self.max_len = int((self.ends - self.starts).max())
        self.pct, _ = lib._elo_pct_array(df)
        self.scale = lib._elo_section_scale(df, self.n, 60.0)
        self.emp = lib._event_mp_limit_scale_array(df, self.n)
        self.ids = {s: df[f"Player_ID_{s}"] for s in "NESW"}
        # Global-stdev seeds: production reads them from the previous run.
        self.gsd_ns = float(df["Elo_R_N_Before"].drop_nulls().drop_nans().std(ddof=0))
        self.gsd_ew = float(df["Elo_R_E_Before"].drop_nulls().drop_nans().std(ddof=0))
        self.slots = {}
        for merged in (False, True):
            if merged:
                (a, b, c, d), n_slots, null = lib._elo_dense_ids([self.ids[s] for s in "NSEW"])
                nulls = [null] if null >= 0 else []
            else:
                (a, b), n_ns, null_ns = lib._elo_dense_ids([self.ids["N"], self.ids["S"]])
                (c, d), n_ew, null_ew = lib._elo_dense_ids([self.ids["E"], self.ids["W"]])
                c = c + n_ns
                d = d + n_ns
                n_slots = n_ns + n_ew
                nulls = ([null_ns] if null_ns >= 0 else []) + ([null_ew + n_ns] if null_ew >= 0 else [])
            self.slots[merged] = (a, b, c, d, n_slots, np.array(nulls, dtype=np.int64))

    def first_values(self, merged: bool, values: dict[str, np.ndarray]) -> np.ndarray:
        """Per slot: the value on the slot's first row in processing order (NaN if none)."""
        a, b, c, d, n_slots, _ = self.slots[merged]
        rank = np.empty(self.n, dtype=np.int64)
        rank[self.proc_order] = np.arange(self.n)
        frames = [
            pl.DataFrame({"slot": idx, "rank": rank, "v": values[s]})
            for s, idx in zip("NSEW", (a, b, c, d))
        ]
        first = (
            pl.concat(frames).sort("rank").group_by("slot", maintain_order=True).first()
        )
        out = np.full(n_slots, np.nan, dtype=np.float64)
        out[first["slot"].to_numpy()] = first["v"].cast(pl.Float64).to_numpy()
        return out


def replay(
    lay: Layout,
    *,
    merged: bool = False,
    k_base: float = 24.0,
    boost: float = 1.5,
    boost_until: int = 100,
    elo_scale: float = 400.0,
    kmul: np.ndarray | None = None,
    init: np.ndarray | None = None,
    pop_mean: float = 1500.0,
    anchor: dict[str, np.ndarray] | None = None,
    alpha: float = 0.0,
) -> pl.DataFrame:
    a, b, c, d, n_slots, nulls = lay.slots[merged]
    init_arr = np.full(n_slots, 1500.0, dtype=np.float32) if init is None else init.astype(np.float32)
    km = lay.emp.astype(np.float32) if kmul is None else (lay.emp * kmul).astype(np.float32)
    out = {k: np.empty(lay.n, dtype=np.float32) for k in ("r_N", "r_S", "r_E", "r_W")}
    cnt = {k: np.empty(lay.n, dtype=np.int32) for k in ("n_N", "n_S", "n_E", "n_W")}
    end = {k: np.empty(lay.n, dtype=np.float32) for k in ("x_N", "x_S", "x_E", "x_W")}
    if anchor is None:
        anchor = {x: np.full(lay.n, np.nan, dtype=np.float32) for x in "NSEW"}
    anc = {x: np.asarray(anchor[x], dtype=np.float32) for x in "NSEW"}
    _kernel(lay.proc_order, lay.starts, lay.ends, lay.board, a, b, c, d,
            lay.pct, lay.scale, km, init_arr, anc["N"], anc["S"], anc["E"], anc["W"], float(alpha), nulls, lay.max_len, float(pop_mean), float(k_base),
            float(boost), int(boost_until), float(elo_scale), 200.0, 3000.0,
            lay.gsd_ns, lay.gsd_ew, lib._elo_anchors(),
            out["r_N"], out["r_S"], out["r_E"], out["r_W"],
            cnt["n_N"], cnt["n_S"], cnt["n_E"], cnt["n_W"],
            end["x_N"], end["x_S"], end["x_E"], end["x_W"])
    return pl.DataFrame({**out, **cnt, **end})
