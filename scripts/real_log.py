"""
real_log.py — 実機ログ（jetson_project の run_signal_playback.py の出力）の読み込みと、指令の時間軸への位置合わせ

Isaac Lab に依存しない（numpy / pandas だけ）。replay_open_loop.py の measured モードと
analysis/eval/compare_replay.py が同じ処理を使う。

- 実機ログは DF/F の圧力列が入れ替わっている（2026/9/29 判明）：物理 DF = meas_pres_F、物理 F = meas_pres_DF
- 時刻は flag 列（MicroLabBox が受け取った DF 指令のエコー）を指令 DF に合わせて揃える
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REAL_DT = 0.005   # 実機ログの刻み [s]
CMD_DT = 0.02     # 指令（test_signals/tm_*.csv）の刻み [s]

# 物理 DF / F / G の順に並べた実機ログの圧力列（入れ替わりを戻す）
PRES_COLS = ("meas_pres_F", "meas_pres_DF", "meas_pres_G")


def echo_lag(flag, cmd_df_50hz, search=(-100, 400)):
    """実機ログ上で flag（DF 指令のエコー）が指令 DF より何行ずれているか（行数, 1 行 = REAL_DT）"""
    flag = np.nan_to_num(np.asarray(flag, float))
    t_real = np.arange(len(flag)) * REAL_DT
    # 指令DF（50 Hz, ZOH）を 200 Hz に展開
    cmd_df = np.asarray(cmd_df_50hz)[np.clip((t_real / CMD_DT).astype(int), 0, len(cmd_df_50hz) - 1)]
    best = (1e9, 0)
    for lag in range(*search):
        if lag >= 0:
            a = cmd_df[: len(cmd_df) - lag]; b = flag[lag: lag + len(a)]
        else:
            b = flag[: len(flag) + lag]; a = cmd_df[-lag: -lag + len(b)]
        n = min(len(a), len(b))
        if n < 100:
            continue
        e = np.mean((a[:n] - b[:n]) ** 2)
        if e < best[0]:
            best = (e, lag)
    return best[1]


def align_real_log(d, cmd_50hz, dt, columns=PRES_COLS):
    """実機ログ d（DataFrame）の columns を、指令の時間軸（0 から指令の最後まで, dt 刻み）に合わせて返す

    戻り値: (n, len(columns)) の配列, ずれ [行]
    """
    lag = echo_lag(d["flag"].values, np.asarray(cmd_50hz)[:, 0])
    t_real = np.arange(len(d)) * REAL_DT
    t_cmd = np.arange(len(cmd_50hz)) * CMD_DT
    t_sim = np.arange(int(round(t_cmd[-1] / dt)) + 1) * dt
    out = np.stack([np.interp(t_sim + lag * REAL_DT, t_real, d[c].values.astype(float)) for c in columns], 1)
    return out, lag


def load_real_aligned(path, cmd_50hz, dt):
    """実機ログを、エコー（flag）で指令の時間軸に合わせ、dt 刻みの DF/F/G 圧力列にして返す"""
    out, lag = align_real_log(pd.read_csv(path), cmd_50hz, dt)
    print(f"[real] エコーで合わせたずれ: {lag * 5} ms（flag が指令に一致する位置）")
    return out
