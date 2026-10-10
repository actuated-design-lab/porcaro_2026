"""run_hys_sweep.py — ヒステリシス力のモデル（relay / play）を振って、開ループ再生をまとめて回す

2026/10/11 の実験用（JFPS 2026秋季、data/user0/jfps2026/replay/hys/）。scripts/replay_open_loop.py を
1 条件 1 プロセスで順に呼ぶ（Isaac Lab の python で実行すること）。出力が既にある条件は飛ばす。

段階
  stage1（同定）: tm_A_quasistatic を、実測圧力（平滑化・遊びなし）で流す。圧力モデルの誤差を除いて
                  ヒステリシスだけを比べるため。relay（今のモデル）と play（幅 w × 大きさ cP）を振る
  stage2（検証）: tm_B〜E（gmd138 は除く）を、流量上限モデル orificeV2 と実測圧力で、relay と選んだ play で流す

ファイル名: sim_<動作>__<圧力モデル>__<ヒステリシス>.{csv,json}
  圧力モデル: measured_raw（実測圧力そのまま）, measured_v2（平滑化 50 ms＋遊び ±10 kPa。10/9 の measuredV2 と同じ）,
              orificeV2（流量上限、shaped の土台と同じ O パラメータ）
  ヒステリシス: relay, play_w<幅kPa>_cp<cP>（例 play_w20_cp15）

使い方:
  python analysis/eval/run_hys_sweep.py --stage 1
  python analysis/eval/run_hys_sweep.py --stage 2 --play_w 0.02 --play_cp 15     # stage1 の結果で選んだ値
  python analysis/eval/run_hys_sweep.py --stage 1 --dry_run                       # コマンドだけ表示
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import real_log as RL  # noqa: E402

O_PARAMS = {"tau": 0.0833, "L": 0.045, "c_in": 4.8542, "c_out": 9.962, "ps": 0.5776, "b": 0.1323}
STAGE1_MOTIONS = ["tm_A_quasistatic"]
STAGE2_MOTIONS = ["tm_B_steps", "tm_C_sine", "tm_D_antagonist", "tm_E_dbl160_seed2", "tm_E_dbl160_seed3"]
STAGE1_W = [0.01, 0.02, 0.04]            # 片側の幅 w [MPa]
STAGE1_CP = [15.0, 7.5]                  # 大きさ cP [N/MPa]（c0 = 0.5 N は固定）


def hys_variants(stage, args):
    v = [("relay", {"pam_hys_mode": "relay"})]
    if stage == 1:
        for w, cp in itertools.product(STAGE1_W, STAGE1_CP):
            v.append((f"play_w{w * 1000:.0f}_cp{cp:g}",
                      {"pam_hys_mode": "play", "pam_hys_play_widths": [w], "pam_hys_coef_p": cp}))
    else:
        w, cp = args.play_w, args.play_cp
        v.append((f"play_w{w * 1000:.0f}_cp{cp:g}",
                  {"pam_hys_mode": "play", "pam_hys_play_widths": [w], "pam_hys_coef_p": cp}))
    return v


def pmodel_args(pm, jetson, motion):
    if pm == "orificeV2":
        return ["--pmodel", "orifice", "--params", json.dumps(O_PARAMS)]
    log = RL.find_real_log(jetson, motion)
    if pm == "measured_raw":
        return ["--pmodel", "measured", "--real_log", log, "--meas_lpf_ms", "0", "--meas_play_kpa", "0"]
    if pm == "measured_v2":
        return ["--pmodel", "measured", "--real_log", log]
    raise ValueError(pm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "data/user0/jfps2026/replay/hys"))
    p.add_argument("--play_w", type=float, default=0.02, help="stage2 で使う play の片側の幅 [MPa]")
    p.add_argument("--play_cp", type=float, default=15.0, help="stage2 で使う cP [N/MPa]")
    p.add_argument("--motions", nargs="*", default=None, help="動作を絞る（既定は段階ごとの一覧）")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    motions = a.motions or (STAGE1_MOTIONS if a.stage == 1 else STAGE2_MOTIONS)
    pmodels = ["measured_raw"] + (["measured_v2"] if a.stage == 1 else ["orificeV2"])
    jobs = []
    for m, pm, (tag, ctrl) in itertools.product(motions, pmodels, hys_variants(a.stage, a)):
        if a.stage == 1 and pm == "measured_v2" and tag != "relay":
            continue                                      # measured_v2 は今の運用（relay）の参照としてだけ流す
        out = os.path.join(a.out_dir, f"sim_{m}__{pm}__{tag}.csv")
        cmd = [sys.executable, os.path.join(ROOT, "scripts/replay_open_loop.py"),
               "--signal", RL.jetson_signal_path(a.jetson, m), *pmodel_args(pm, a.jetson, m),
               "--ctrl", json.dumps(ctrl), "--no_drum", "--headless", "--out", out]
        jobs.append((out, cmd))
    print(f"[sweep] stage {a.stage}: {len(jobs)} 条件（既にあるものは飛ばす）")
    for i, (out, cmd) in enumerate(jobs, 1):
        if os.path.exists(out):
            print(f"  [{i}/{len(jobs)}] skip {os.path.basename(out)}"); continue
        print(f"  [{i}/{len(jobs)}] {os.path.basename(out)}")
        if a.dry_run:
            print("    " + " ".join(cmd)); continue
        t = time.time(); r = subprocess.run(cmd, cwd=ROOT)
        print(f"    rc={r.returncode} ({time.time() - t:.0f} s)")
        if r.returncode != 0:
            print("    失敗。ここで止める"); sys.exit(r.returncode)


if __name__ == "__main__":
    main()
