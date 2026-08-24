"""
aggregate_tau_strikes.py — τスイープの評価結果を 1N 判定で集計する。

■ 前提
  eval_logs_tau_sweep/ に既に 81ジョブ（27run × 3条件）の
  simulation_log.csv が書き出されている（2026-07-20 実行済み）。
  ここでは **GPUを使わず、そのCSVを読み直すだけ**。

■ なぜ 1N か
  学習時に力のハード閾値は無い（rewards_cfg は target_force_fd=20 / sigma_force=15
  のガウス型ソフト評価）。閾値は評価時の選択にすぎず、
  本体eval・実機と揃えるために 1N（min_strike_frac=0.05）を使う。

■ 出力
  eval_logs_tau_sweep/tau_summary_1N.csv … run×条件ごとの success_rate 等
  eval_logs_tau_sweep/tau_strikes_1N.csv … 打点ごと（誤差分布用）
  さらに標準出力に「τ × lookahead」のピボット表を出す（Fig.6 のもと）

■ run ディレクトリ名の規約（Claude Code の調査結果より）
  ..._tau{X}_lh{Y}_seed{N}    例: 2026-07-14_16-43-00_tau0.5_lh0.1_seed1
  ※ 自己完結にするため run_eval_tau_sweep.py には依存しない（名前から直接読む）。
    もし env.yaml と食い違う場合は --verify_yaml で突き合わせできる。

Usage:
  python analysis/eval/aggregate_tau_strikes.py --dry_run
  python analysis/eval/aggregate_tau_strikes.py
  python analysis/eval/aggregate_tau_strikes.py --verify_yaml   # env.yaml と照合
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.harness.strike_extract import extract_strikes  # noqa: E402
# isaaclab の dump_yaml は python/tuple や slice のタグを吐くため、
# 素の yaml.safe_load では読めない。リポジトリ既存の許可リスト付きローダーを使う。
from analysis.harness.identify import load_yaml  # noqa: E402

MIN_STRIKE_FRAC_1N = 0.05      # x target_ref(20N) = 1 N
TARGET_REF = 20.0
TOL_MS = 30.0

TAU_RUN_RE = re.compile(r"_tau([0-9.]+)_lh([0-9.]+)_seed(\d+)$")


def parse_run_tag(tag: str):
    m = TAU_RUN_RE.search(tag)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def bpm_from_tag(tag: str):
    m = re.search(r"_(\d+)bpm", tag) or re.search(r"bpm(\d+)", tag)
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_logs", default=str(REPO_ROOT / "eval_logs_tau_sweep"))
    ap.add_argument("--logs_root", default=str(REPO_ROOT / "logs" / "rsl_rl_tau_sweep"))
    ap.add_argument("--min_strike_frac", type=float, default=MIN_STRIKE_FRAC_1N)
    ap.add_argument("--tol_ms", type=float, default=TOL_MS)
    ap.add_argument("--verify_yaml", action="store_true",
                    help="params/env.yaml の pam_tau_scale_range / lookahead_horizon と照合")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.eval_logs).resolve()
    if not root.exists():
        print(f"ERROR: {root} が無い。--eval_logs で指定してください。")
        return 1

    logs = sorted(root.rglob("simulation_log.csv"))
    print(f"[info] simulation_log.csv: {len(logs)} 件  ({root})")
    if not logs:
        return 1

    # --- 任意: env.yaml と照合して命名を信用してよいか確かめる ---
    if args.verify_yaml:
        bad = 0
        for d in sorted(Path(args.logs_root).rglob("params/env.yaml")):
            tag = d.parent.parent.name
            p = parse_run_tag(tag)
            if p is None:
                continue
            y = load_yaml(d)
            rng = y.get("pam_tau_scale_range")
            lh = y.get("lookahead_horizon")
            tau_y = rng[0] if isinstance(rng, (list, tuple)) else rng
            if abs(float(tau_y) - p[0]) > 1e-6 or abs(float(lh) - p[1]) > 1e-6:
                print(f"  [MISMATCH] {tag}: 名前(tau={p[0]}, lh={p[1]}) vs "
                      f"yaml(tau={tau_y}, lh={lh})")
                bad += 1
        print(f"[verify] 不一致 {bad} 件\n")

    rows, strikes, skipped = [], [], []
    for p in logs:
        cond_tag = p.parent.name
        run_tag = p.parent.parent.parent.name
        parsed = parse_run_tag(run_tag)
        if parsed is None:
            skipped.append((run_tag, "run名から tau/lh/seed が読めない")); continue
        tau, lh, seed = parsed

        bpm = bpm_from_tag(cond_tag)
        if bpm is None:
            skipped.append((cond_tag, "BPMが読めない")); continue

        try:
            ts = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001
            skipped.append((str(p), f"読めない: {e}")); continue
        if not {"time_s", "target_force", "force_z"}.issubset(ts.columns):
            skipped.append((str(p), "列不足")); continue
        if len(ts) < 10:
            skipped.append((str(p), f"行数 {len(ts)}（失敗ジョブの可能性）")); continue

        cond = re.sub(r"_trial\d+$", "", cond_tag)
        if args.dry_run:
            rows.append(dict(tau=tau, lh=lh, seed=seed, task=cond, n_rows=len(ts)))
            continue

        st = extract_strikes(ts, bpm=bpm, target_ref=TARGET_REF, tol_ms=args.tol_ms,
                             min_strike_frac=args.min_strike_frac)
        if st.empty:
            skipped.append((str(p), "打点なし")); continue
        thr = args.min_strike_frac * TARGET_REF
        hit = st[st["peak_force"] >= thr]

        s2 = st.copy()
        s2["tau"], s2["lh"], s2["seed"] = tau, lh, seed
        s2["task"], s2["bpm"], s2["source"] = cond, bpm, "sim_tau"
        strikes.append(s2)

        rows.append(dict(
            tau=tau, lh=lh, seed=seed, task=cond, bpm=bpm, n_strikes=len(st),
            success_rate=float(st["success"].mean()),
            abs_err_ms_mean=float(hit["timing_err_ms"].abs().mean()) if len(hit) else np.nan,
            err_ms_mean=float(hit["timing_err_ms"].mean()) if len(hit) else np.nan,
            err_ms_std=float(hit["timing_err_ms"].std()) if len(hit) else np.nan,
            peak_force_mean=float(st["peak_force"].mean()),
            miss_rate_force=float((st["peak_force"] < thr).mean()),
            force_threshold_N=thr, file=str(p.relative_to(REPO_ROOT)),
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("集計対象なし。skipped:")
        for s in skipped[:20]:
            print("  ", s)
        return 1

    print("\n=== 見つかったセル（τ × lookahead、値は seed数）===")
    print(df.pivot_table(index="tau", columns="lh", values="seed",
                         aggfunc="nunique").fillna(0).astype(int).to_string())
    if skipped:
        print(f"\n[skip] {len(skipped)} 件"); [print("  ", s) for s in skipped[:8]]

    if args.dry_run:
        print("\n=== --dry_run のため書き出していません ===")
        return 0

    out1 = root / "tau_summary_1N.csv"
    out2 = root / "tau_strikes_1N.csv"
    df.to_csv(out1, index=False)
    pd.concat(strikes, ignore_index=True).to_csv(out2, index=False)
    print(f"\n[saved] {out1}  ({len(df)} ラン)")
    print(f"[saved] {out2}  ({sum(len(s) for s in strikes)} 打点)")

    # --- Fig.6 のもと：τ × lookahead の成功率（seed平均）---
    print("\n=== ★ τ × lookahead の ±30ms成功率（全条件平均、seed平均）===")
    seed_lv = df.groupby(["tau", "lh", "seed"]).success_rate.mean().reset_index()
    pv = seed_lv.pivot_table(index="tau", columns="lh", values="success_rate", aggfunc="mean")
    print(pv.round(3).to_string())
    print("\n  各τでの最良 lookahead:")
    for tau in sorted(seed_lv.tau.unique()):
        x = seed_lv[seed_lv.tau == tau].groupby("lh").success_rate.agg(["mean", "std", "size"])
        best = x["mean"].idxmax()
        print(f"    tau={tau}: 最良 lh={best:.2f}s  ({x.loc[best,'mean']:.3f} "
              f"± {x.loc[best,'std']:.3f}, n={int(x.loc[best,'size'])})   "
              f"全水準 {dict(x['mean'].round(3))}")

    print("\n=== 複雑リズム(GMD)のみ ===")
    gmd = seed_lv.merge(df[["tau", "lh", "seed", "task"]].drop_duplicates(),
                        on=["tau", "lh", "seed"], how="left") if "task" not in seed_lv else seed_lv
    g = df[df.task.str.startswith("gmd")].groupby(["tau", "lh", "seed"]).success_rate.mean().reset_index()
    print(g.pivot_table(index="tau", columns="lh", values="success_rate", aggfunc="mean")
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
