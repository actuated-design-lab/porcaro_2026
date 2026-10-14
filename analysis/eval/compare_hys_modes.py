"""compare_hys_modes.py — PAM 摩擦の向きの決め方（play＝圧力 / play_length＝筋の長さ）で再生した sim を、実機と手首角で比べる

入力: --dir 以下の sim_<動作>__<タグ>.csv（replay_open_loop.py の出力。サブフォルダも探す）と、同名の .json（real_log の場所）
      例: sim_tm_B_steps__play_p.csv、static_len003/sim_id_static_cocontraction__len003.csv
指標（手首角、最初の --skip_s 秒は除く）:
  RMSE、平均ずれ（sim − 実機）、止まっているとき（実機の角速度 < 5 deg/s）の RMSE、動いているとき（> 50 deg/s）の RMSE
出力: <dir>/summary.md（動作 × タグの表）、<dir>/metrics.csv

使い方:
  python analysis/eval/compare_hys_modes.py --dir data/user0/jfps2026/replay/hys_length
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools", "user0", "jfps2026"))
import real_log as RL  # noqa: E402

DT = 0.005


def load_real(jetson, motion, real_log):
    cmd = pd.read_csv(RL.jetson_signal_path(jetson, motion))
    c3 = cmd[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values
    path = real_log if real_log and os.path.exists(real_log) else RL.find_real_log(jetson, motion, os.path.basename(real_log or "") or None)
    A, _ = RL.align_real_log(pd.read_csv(path), c3, DT, columns=RL.PRES_COLS + ("wrist_angle_deg",))
    return A[:, 3], path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    ap.add_argument("--skip_s", type=float, default=2.0)
    a = ap.parse_args()
    rows, cache = [], {}
    for f in sorted(glob.glob(os.path.join(a.dir, "**", "sim_*__*.csv"), recursive=True)):
        name = os.path.basename(f)[4:-4]
        motion, tag = name.split("__", 1)
        meta = os.path.splitext(f)[0] + ".json"
        rl = json.load(open(meta, encoding="utf-8")).get("real_log") if os.path.exists(meta) else None
        if motion not in cache:
            cache[motion] = load_real(a.jetson, motion, rl)
        real, path = cache[motion]
        s = pd.read_csv(f).wrist_angle_deg.values
        n = min(len(s), len(real)); i0 = int(a.skip_s / DT)
        e = (s[:n] - real[:n])[i0:]
        w = np.abs(np.gradient(np.convolve(real[:n], np.ones(9) / 9, "same"), DT))[i0:]
        rest, mov = w < 5, w > 50
        rm = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan
        rows.append(dict(motion=motion, tag=tag, rmse=rm(e), offset=float(e.mean()), rmse_rest=rm(e[rest]),
                         rmse_moving=rm(e[mov]), rest_pct=100 * rest.mean(), moving_pct=100 * mov.mean(),
                         real_log=os.path.basename(path)))
    M = pd.DataFrame(rows)
    os.makedirs(a.dir, exist_ok=True)
    M.to_csv(os.path.join(a.dir, "metrics.csv"), index=False, float_format="%.3f")
    lines = ["# PAM 摩擦の向きの決め方の比較（compare_hys_modes.py）", "",
             f"手首角 [deg]、最初の {a.skip_s:g} s を除く。止まっている＝実機の角速度 < 5 deg/s、動いている＝ > 50 deg/s", ""]
    for col, lab in (("rmse", "RMSE"), ("rmse_rest", "止まっているときの RMSE"), ("rmse_moving", "動いているときの RMSE"),
                     ("offset", "平均ずれ（sim − 実機）")):
        lines += [f"## {lab}", "", "```", M.pivot(index="motion", columns="tag", values=col).round(2).to_string(), "```", ""]
    open(os.path.join(a.dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
