"""
Tier1実験(Step1: LSTM lookahead sweep, Step2: MLP / Frame-stacking MLP)を
Nシード分まとめて自動実行するループスクリプト。

配置場所: scripts/rsl_rl/run_experiment_matrix.py
         (train.py, cli_args.py と同じディレクトリに置くこと)

使い方:
  # まずはコマンド一覧だけ確認 (実行はしない)
  python scripts/rsl_rl/run_experiment_matrix.py --dry_run

  # 実際に実行 (順番に1本ずつ、GPU1枚前提なので並列化はしない)
  python scripts/rsl_rl/run_experiment_matrix.py

  # シード数やmax_iterationsを変えたい場合
  python scripts/rsl_rl/run_experiment_matrix.py --num_seeds 5 --max_iterations 1500
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ==============================================================================
# 実験条件の定義
# ==============================================================================
# ここに条件を足せば、それだけでループ対象が増える。
# exp_tag は experiment_name の生成に使う「人間が読める短い識別子」。
# 実際にtrain.pyへ渡す値(lookahead_horizon等)は別途明示的に持たせ、
# exp_tag文字列とは独立させることで「名前と実体がズレる」事故を防ぐ。

def build_conditions() -> list[dict]:
    conditions = []

    # --- Step1: LSTM lookahead sweep ---
    for lh in [0.1, 0.5, 1.0]:
        conditions.append({
            "agent": "rsl_rl_lstm_cfg_entry_point",
            "lookahead_horizon": lh,
            "use_frame_stacking": False,
            "frame_stack_k": None,
            "exp_tag": f"lstm_lookahead{lh}",
        })

    # --- Step2: MLP (記憶なし, LSTMのlookahead=0.5と対で比較する条件) ---
    conditions.append({
        "agent": "rsl_rl_mlp_cfg_entry_point",
        "lookahead_horizon": 0.5,
        "use_frame_stacking": False,
        "frame_stack_k": None,
        "exp_tag": "mlp_plain_lookahead0.5",
    })

    # --- Step2: Frame-stacking MLP (有限記憶, k=5) ---
    conditions.append({
        "agent": "rsl_rl_mlp_cfg_entry_point",
        "lookahead_horizon": 0.5,
        "use_frame_stacking": True,
        "frame_stack_k": 5,
        "exp_tag": "mlp_framestack_k5_lookahead0.5",
    })

    return conditions


# ==============================================================================
# 実験名の自動生成 (手打ちを排除する)
# ==============================================================================

def make_experiment_name(cond: dict) -> str:
    return f"porcaro_{cond['exp_tag']}"


def make_run_name(seed: int) -> str:
    return f"seed{seed}"


# ==============================================================================
# 完了判定 (再実行時に、既に終わっている条件はスキップする)
# ==============================================================================

def is_already_done(logs_root: str, experiment_name: str, run_name: str, max_iterations: int) -> str | None:
    """
    logs/rsl_rl/{experiment_name}/ 配下から run_name に一致するフォルダを探し、
    最終チェックポイント(model_{max_iterations-1}.pt など)が存在するか確認する。
    見つかればそのパスを返す。見つからなければ None。
    """
    exp_dir = os.path.join(logs_root, experiment_name)
    if not os.path.isdir(exp_dir):
        return None

    for d in sorted(os.listdir(exp_dir)):
        if d.endswith(f"_{run_name}") or d == run_name:
            run_dir = os.path.join(exp_dir, d)
            if not os.path.isdir(run_dir):
                continue
            # save_interval=50 保存の場合、ちょうど max_iterations-1 とは限らないため、
            # 「一定以上のiteration番号のptファイルが存在するか」で判定する。
            ckpts = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
            for ckpt in ckpts:
                try:
                    num = int(ckpt.replace("model_", "").replace(".pt", ""))
                except ValueError:
                    continue
                # 最終iterationの直前(save_interval分の誤差)まで進んでいれば完了とみなす
                if num >= max_iterations - 51:
                    return os.path.join(run_dir, ckpt)
    return None


# ==============================================================================
# コマンド組み立て
# ==============================================================================

def build_command(cond: dict, seed: int, args: argparse.Namespace) -> tuple[list[str], str, str]:
    experiment_name = make_experiment_name(cond)
    run_name = make_run_name(seed)

    cmd = [
        sys.executable, "scripts/rsl_rl/train.py",
        "--task", args.task,
        "--agent", cond["agent"],
        "--seed", str(seed),
        "--lookahead_horizon", str(cond["lookahead_horizon"]),
        "--experiment_name", experiment_name,
        "--run_name", run_name,
        "--max_iterations", str(args.max_iterations),
        "--headless",
    ]

    if cond["use_frame_stacking"]:
        cmd += ["--use_frame_stacking", "--frame_stack_k", str(cond["frame_stack_k"])]

    return cmd, experiment_name, run_name


# ==============================================================================
# メイン処理
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier1実験の一括実行ループ")
    parser.add_argument("--task", type=str, default="Template-Porcaro-2026-ModelB-DR-user0",
                        help="対象タスクID")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="各条件ごとのシード本数 (1..num_seeds を使用)")
    parser.add_argument("--max_iterations", type=int, default=1500,
                        help="1本あたりの学習イテレーション数")
    parser.add_argument("--logs_root", type=str, default="logs/rsl_rl",
                        help="ログのルートディレクトリ (完了判定に使用)")
    parser.add_argument("--dry_run", action="store_true",
                        help="実行はせず、実行予定のコマンド一覧だけ表示する")
    parser.add_argument("--manifest", type=str, default="logs/experiment_matrix_manifest.json",
                        help="実行結果を記録するJSONファイルのパス")
    args = parser.parse_args()

    conditions = build_conditions()
    seeds = list(range(1, args.num_seeds + 1))

    # 実行計画を全部書き出す
    plan = []
    for cond in conditions:
        for seed in seeds:
            cmd, experiment_name, run_name = build_command(cond, seed, args)
            plan.append({
                "cond": cond,
                "seed": seed,
                "cmd": cmd,
                "experiment_name": experiment_name,
                "run_name": run_name,
            })

    print(f"[Plan] 合計 {len(plan)} 本の学習run (条件{len(conditions)} × シード{len(seeds)})")
    for i, item in enumerate(plan):
        print(f"  [{i+1:02d}] {item['experiment_name']} / {item['run_name']}  "
              f"(agent={item['cond']['agent']}, lookahead={item['cond']['lookahead_horizon']}, "
              f"frame_stack={item['cond']['use_frame_stacking']})")

    if args.dry_run:
        print("\n[Dry Run] 実行はしていません。--dry_run を外すと実際に実行されます。")
        return

    # マニフェストの読み込み(前回までの実行記録があれば引き継ぐ)
    manifest = {}
    if os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    for i, item in enumerate(plan):
        key = f"{item['experiment_name']}/{item['run_name']}"
        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(plan)}] {key}")
        print("=" * 80)

        # --- 既に完了しているかチェック(再開機能) ---
        done_ckpt = is_already_done(args.logs_root, item["experiment_name"], item["run_name"], args.max_iterations)
        if done_ckpt is not None:
            print(f"[Skip] 既に完了済みと判定: {done_ckpt}")
            manifest[key] = {"status": "skipped_already_done", "checkpoint": done_ckpt}
            continue

        print("[Command]", " ".join(item["cmd"]))
        start_time = time.time()
        manifest[key] = {"status": "running", "started_at": datetime.now().isoformat()}
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        try:
            result = subprocess.run(item["cmd"])
            elapsed = time.time() - start_time
            if result.returncode == 0:
                print(f"[Success] {key} ({elapsed/60:.1f} min)")
                manifest[key] = {
                    "status": "success",
                    "elapsed_min": round(elapsed / 60, 1),
                    "finished_at": datetime.now().isoformat(),
                }
            else:
                print(f"[Failed] {key} (returncode={result.returncode})")
                manifest[key] = {
                    "status": "failed",
                    "returncode": result.returncode,
                    "finished_at": datetime.now().isoformat(),
                }
        except Exception as e:
            print(f"[Error] {key}: {e}")
            manifest[key] = {"status": "error", "error": str(e)}

        # --- 1本ごとにマニフェストを保存(途中で止まっても記録が残る) ---
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[All Done] 全run終了。マニフェスト:", args.manifest)


if __name__ == "__main__":
    main()
