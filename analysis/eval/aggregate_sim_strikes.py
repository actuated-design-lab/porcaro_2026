"""
aggregate_sim_strikes.py — sim の simulation_log.csv を 1N 判定で再集計し、
                           実機と同じ形式のCSVを2本出す。

■ なぜ 1N なのか
  学習時に力のハード閾値は存在しない（user0/cfg/rewards_cfg.py の力の報酬は
  target_force_fd=20.0 / sigma_force=15.0 のガウス型ソフト評価）。
  つまり「何Nで打撃と認めるか」は評価時の選択にすぎず、再学習は不要。
  IROS時の analysis/create_fig7.py は FORCE_THRESHOLD=1.0 だったため 1N に揃える。
  → strike_extract.extract_strikes(min_strike_frac=0.05) が 0.05 x 20N = 1N。

■ GPUは使わない
  eval_logs/ に既に書き出されている simulation_log.csv を読むだけ。
  isaaclab / torch は import しない（pandas + numpy + scipy のみ）。

Usage:
  # まず何が見つかるか確認（何も書き出さない）
  python analysis/eval/aggregate_sim_strikes.py --dry_run

  # 本番
  python analysis/eval/aggregate_sim_strikes.py

出力:
  eval_logs/sim_summary_1N.csv  … model, seed, task, trial, success_rate,
                                   abs_err_ms_mean, err_ms_mean, err_ms_std,
                                   peak_force_mean, miss_rate_force, n_strikes
  eval_logs/sim_strikes_1N.csv  … 打点ごと（timing_err_ms, peak_force, success 等）
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.harness.discover import discover_all_runs  # noqa: E402
from analysis.harness.strike_extract import extract_strikes  # noqa: E402

MIN_STRIKE_FRAC_1N = 0.05     # 0.05 x target_ref(20N) = 1 N
TARGET_REF = 20.0
TOL_MS = 30.0

# 実機で取得した3条件。これに合わせると直接比較できる。
REAL_CONDITIONS = {
    "single_8_120bpm": "test_single8_bpm120",
    "double_160bpm": "test_double_bpm160",
    "gmd_03_high_bpm138": "gmd_03_high_bpm138",
}


def bpm_from_tag(tag: str) -> float | None:
    """condition_tag から BPM を取り出す。
    play_sim_rhythm.py : '{pattern}_{bpm}bpm_trial{n}'  例 double_160bpm_trial0
    play_sim_midi.py   : '{midi_stem}_trial{n}'         例 gmd_03_high_bpm138_trial0
    """
    m = re.search(r"_(\d+)bpm", tag) or re.search(r"bpm(\d+)", tag)
    return float(m.group(1)) if m else None


def trial_from_tag(tag: str) -> int:
    m = re.search(r"trial(\d+)", tag)
    return int(m.group(1)) if m else 0


def condition_key(tag: str) -> str:
    """trial番号を落として条件名だけにする。"""
    return re.sub(r"_trial\d+$", "", tag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_logs", default=str(REPO_ROOT / "eval_logs"))
    ap.add_argument("--min_strike_frac", type=float, default=MIN_STRIKE_FRAC_1N,
                    help="打撃と認める最小の力 / target_ref。既定 0.05 (=1N)")
    ap.add_argument("--tol_ms", type=float, default=TOL_MS)
    ap.add_argument("--only_real_conditions", action="store_true",
                    help="実機で取得した3条件だけに絞る")
    ap.add_argument("--dry_run", action="store_true", help="集計せず、見つかったものを表示")
    args = ap.parse_args()

    eval_root = Path(args.eval_logs).resolve()
    if not eval_root.exists():
        print(f"ERROR: {eval_root} が無い")
        return 1

    # --- run_dir名 -> (model, seed) の対応表を discover から作る ---
    runs = discover_all_runs(str(REPO_ROOT / "logs" / "rsl_rl"))
    tag2ms = {Path(r.run_dir).name: (r.model, r.seed) for r in runs.itertuples()}
    print(f"[info] discover: {len(tag2ms)} run を認識")

    logs = sorted(eval_root.rglob("simulation_log.csv"))
    print(f"[info] simulation_log.csv: {len(logs)} 件\n")
    if not logs:
        print("ERROR: simulation_log.csv が1件も無い。--eval_logs を確認。")
        return 1

    rows, strikes, skipped = [], [], []
    for p in logs:
        # eval_logs/{run_tag}/{ckpt}/{condition_tag}/simulation_log.csv
        try:
            cond_tag = p.parent.name
            ckpt = p.parent.parent.name
            run_tag = p.parent.parent.parent.name
        except Exception:  # noqa: BLE001
            skipped.append((str(p), "パス構造が想定外")); continue

        if run_tag not in tag2ms:
            skipped.append((str(p), f"run_tag '{run_tag}' が discover に無い")); continue
        model, seed = tag2ms[run_tag]

        cond = condition_key(cond_tag)
        if args.only_real_conditions and cond not in REAL_CONDITIONS:
            continue

        bpm = bpm_from_tag(cond_tag)
        if bpm is None:
            skipped.append((str(p), f"BPMが読めない: {cond_tag}")); continue

        try:
            ts = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001
            skipped.append((str(p), f"読めない: {e}")); continue

        need = {"time_s", "target_force", "force_z"}
        if not need.issubset(ts.columns):
            skipped.append((str(p), f"列不足: {need - set(ts.columns)}")); continue

        if args.dry_run:
            rows.append(dict(model=model, seed=seed, task=cond,
                             trial=trial_from_tag(cond_tag), bpm=bpm, n_rows=len(ts)))
            continue

        # ★ play_sim_rhythm.py 系のログ（double_160bpm）は 1ファイルに複数エピソードが
        #   連結されており、time_s が各エピソードの先頭で 0 に戻る。そのまま処理すると
        #   (a) find_peaks がエピソード境界をまたぎ、(b) 時刻が重複するため
        #   エピソード1の目標がエピソード2/3の打撃とマッチしうる（候補が3倍になる）。
        #   時刻が戻る点で分割し、エピソードごとに独立に集計して平均する。
        #   play_sim_midi.py 系（GMD）は1エピソードなので分割は起きない。
        seg_bounds = np.r_[0, np.nonzero(np.diff(ts["time_s"].to_numpy(float)) < 0)[0] + 1,
                           len(ts)]
        segs = [ts.iloc[a:b] for a, b in zip(seg_bounds[:-1], seg_bounds[1:]) if b - a > 10]
        sts = []
        for seg in segs:
            try:
                one = extract_strikes(seg.reset_index(drop=True), bpm=bpm,
                                      target_ref=TARGET_REF, tol_ms=args.tol_ms,
                                      min_strike_frac=args.min_strike_frac)
            except Exception:  # noqa: BLE001
                continue
            if not one.empty:
                sts.append(one)
        if not sts:
            skipped.append((str(p), "打点が検出されない")); continue
        n_ep = len(sts)
        st = pd.concat(sts, ignore_index=True)

        thr = args.min_strike_frac * TARGET_REF
        hit = st[st["peak_force"] >= thr]

        s2 = st.copy()
        s2["model"], s2["seed"] = model, seed
        s2["task"], s2["eval_trial"] = cond, trial_from_tag(cond_tag)
        s2["bpm"], s2["source"] = bpm, "sim"
        strikes.append(s2)

        rows.append(dict(
            model=model, seed=seed, task=cond, trial=trial_from_tag(cond_tag), bpm=bpm,
            n_episodes=n_ep, n_strikes=len(st),
            success_rate=float(st["success"].mean()),
            abs_err_ms_mean=float(hit["timing_err_ms"].abs().mean()) if len(hit) else np.nan,
            err_ms_mean=float(hit["timing_err_ms"].mean()) if len(hit) else np.nan,
            err_ms_std=float(hit["timing_err_ms"].std()) if len(hit) else np.nan,
            peak_force_mean=float(st["peak_force"].mean()),
            miss_rate_force=float((st["peak_force"] < thr).mean()),
            force_threshold_N=thr, ckpt=ckpt, file=str(p.relative_to(REPO_ROOT)),
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("集計対象なし。skipped を確認:")
        for s in skipped[:20]:
            print("  ", s)
        return 1

    print("=== 見つかった (model x seed x task) ===")
    print(df.pivot_table(index="model", columns="task", values="seed",
                         aggfunc="nunique").fillna(0).astype(int).to_string())
    if skipped:
        print(f"\n[skip] {len(skipped)} 件")
        for s in skipped[:10]:
            print("  ", s)

    if args.dry_run:
        print("\n=== --dry_run のため書き出していません ===")
        return 0

    out1 = eval_root / "sim_summary_1N.csv"
    out2 = eval_root / "sim_strikes_1N.csv"
    df.to_csv(out1, index=False)
    pd.concat(strikes, ignore_index=True).to_csv(out2, index=False)
    print(f"\n[saved] {out1}  ({len(df)} ラン)")
    print(f"[saved] {out2}  ({sum(len(s) for s in strikes)} 打点)")

    print("\n=== モデル別 成功率（seed単位に集約）===")
    seed_lv = df.groupby(["model", "seed"]).success_rate.mean()
    for m in sorted(df.model.unique()):
        v = seed_lv.loc[m].values
        print(f"  {m}: {v.mean():.3f} ± {v.std(ddof=1):.3f}   seed別 {np.round(v, 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
