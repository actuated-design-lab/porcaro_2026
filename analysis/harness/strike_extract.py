"""Strike-level extraction from simulation_log.csv-shaped data.

Task A finding (see analysis notes / chat): target_force in
eval_logs/2026-03-01_08-00-23/model_1499/double_120bpm/simulation_log.csv is
TIME-VARYING (a piecewise-constant/staircase sampling of the super-Gaussian
strike kernel, held per control step) - it is NOT a constant column. Its
local maxima (peaks, reaching the target hit force at the kernel center)
reliably mark the intended strike times. Peak detection on target_force is
therefore used to recover ground-truth target strike times, and peak
detection on force_z within a window around each target time recovers the
actual (measured) strike time and force.

No isaaclab / torch imports - pandas / numpy / scipy only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def extract_strikes(
    ts: pd.DataFrame,
    bpm: float | None = None,
    target_ref: float = 20.0,
    tol_ms: float = 30.0,
    min_strike_frac: float = 0.25,
    method: str = "peak_match",
) -> pd.DataFrame:
    """Recover per-strike timing/force outcomes from a simulation_log-shaped trace.

    Args:
        ts: DataFrame with at least columns ["time_s", "target_force", "force_z"].
            If `bpm` is not given, a "target_bpm" column is used instead.
        bpm: tempo used to derive T16th = 15/bpm (s) for windowing/peak-distance.
        target_ref: reference peak force (N) used for the min_strike_frac gate
            and (implicitly) as the height threshold for detecting target peaks.
        tol_ms: perceptual timing tolerance (ms) for the `success` flag.
        min_strike_frac: minimum fraction of target_ref a measured peak must
            reach to count as a genuine strike (rather than a graze).
        method: "peak_match" (default) detects genuine strikes as local maxima
            of the measured force above min_strike_frac*target_ref and assigns
            each target the nearest unused strike, one-to-one. "argmax" is the
            legacy behaviour (take the argmax of the force inside a
            +-1.5*T16th window around each target) and is kept only to
            reproduce pre-2026-08-06 numbers.
            ---
            WHY PEAK_MATCH. The legacy windows of adjacent onsets overlap
            whenever the onsets are closer than 3*T16th, so both onsets take
            the argmax of essentially the same stretch of signal - the SAME
            physical strike gets credited to two different targets and one of
            them is forced to a timing error of one inter-onset interval. On
            the 160 BPM double-stroke pattern (onsets 80-120 ms apart, window
            +-140.6 ms) this hit 47% of adjacent pairs on hardware and 50% in
            simulation, capping the achievable success rate near 0.5 however
            well the policy played. It also made the selection depend on force
            magnitude (argmax picks the strongest peak, not the nearest one),
            which the success criterion never intended. peak_match reproduces
            the scheme used for the IROS figures (create_fig7.py) and is
            insensitive to the search-window width: sweeping it from +-45 ms
            to +-150 ms leaves every success rate unchanged.

    Returns:
        DataFrame [strike_idx, target_time, peak_time, timing_err_ms,
                   peak_force, success], one row per detected target strike.
    """
    required_cols = {"time_s", "target_force", "force_z"}
    missing = required_cols - set(ts.columns)
    if missing:
        raise ValueError(f"extract_strikes: missing required columns {missing}")

    time_s = ts["time_s"].to_numpy(dtype=float)
    target_force = ts["target_force"].to_numpy(dtype=float)
    force_z = ts["force_z"].to_numpy(dtype=float)

    if len(time_s) < 2:
        raise ValueError("extract_strikes: need at least 2 samples")

    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        raise ValueError(f"extract_strikes: non-positive inferred dt={dt}")

    if bpm is None:
        if "target_bpm" not in ts.columns:
            raise ValueError("extract_strikes: bpm not given and no 'target_bpm' column present")
        bpm = float(ts["target_bpm"].median())
    bpm = float(bpm)
    if bpm <= 0:
        raise ValueError(f"extract_strikes: invalid bpm={bpm}")

    t16 = 15.0 / bpm  # seconds, per paper Eq: T16th = 15/BPM

    # 1. Recover target strike times from peaks of target_force.
    distance_samples = max(1, int(round(0.3 * t16 / dt)))
    target_height = min_strike_frac * target_ref
    tgt_peak_idx, _ = find_peaks(target_force, height=target_height, distance=distance_samples)
    target_times = time_s[tgt_peak_idx]

    # 2. For each target time, search a window of +-1.5*T16th for the actual
    #    (measured) force_z peak.
    half_win = 1.5 * t16
    strike_gate = min_strike_frac * target_ref

    if method == "peak_match":
        # 1. 実打撃そのものを検出する（力のしきい値を超える極大）。
        act_idx, _ = find_peaks(force_z, height=strike_gate, distance=distance_samples)
        act_t, act_f = time_s[act_idx], force_z[act_idx]
        # 2. |誤差| の小さい組から貪欲に1対1で割り当てる。
        cand = []
        for ti, t_target in enumerate(target_times):
            for ai in np.nonzero(np.abs(act_t - t_target) <= half_win)[0]:
                cand.append((abs(float(act_t[ai]) - float(t_target)), ti, int(ai)))
        cand.sort()
        taken_t, taken_a = {}, set()
        for _, ti, ai in cand:
            if ti in taken_t or ai in taken_a:
                continue
            taken_t[ti] = ai
            taken_a.add(ai)
        rows = []
        for i, t_target in enumerate(target_times):
            j = taken_t.get(i)
            if j is None:
                rows.append(dict(strike_idx=i, target_time=float(t_target),
                                 peak_time=np.nan, timing_err_ms=np.nan,
                                 peak_force=0.0, success=False))
                continue
            err = (float(act_t[j]) - float(t_target)) * 1000.0
            rows.append(dict(strike_idx=i, target_time=float(t_target),
                             peak_time=float(act_t[j]), timing_err_ms=err,
                             peak_force=float(act_f[j]),
                             success=bool(abs(err) <= tol_ms)))
        return pd.DataFrame(
            rows,
            columns=["strike_idx", "target_time", "peak_time", "timing_err_ms",
                     "peak_force", "success"],
        )

    if method != "argmax":
        raise ValueError(f"extract_strikes: unknown method {method!r}")

    rows = []
    for i, t_target in enumerate(target_times):
        mask = ((time_s >= t_target - half_win)
                & (time_s <= t_target + half_win))
        if not mask.any():
            rows.append(
                dict(
                    strike_idx=i,
                    target_time=float(t_target),
                    peak_time=np.nan,
                    timing_err_ms=np.nan,
                    peak_force=0.0,
                    success=False,
                )
            )
            continue

        local_t = time_s[mask]
        local_f = force_z[mask]
        j = int(np.argmax(local_f))
        peak_force = float(local_f[j])
        peak_time = float(local_t[j])
        timing_err_ms = (peak_time - t_target) * 1000.0
        success = bool(peak_force >= min_strike_frac * target_ref and abs(timing_err_ms) <= tol_ms)

        rows.append(
            dict(
                strike_idx=i,
                target_time=float(t_target),
                peak_time=peak_time,
                timing_err_ms=timing_err_ms,
                peak_force=peak_force,
                success=success,
            )
        )

    return pd.DataFrame(
        rows,
        columns=["strike_idx", "target_time", "peak_time", "timing_err_ms", "peak_force", "success"],
    )
