"""
export_onnx_matrix.py — 完了済み全run(model x seed)のONNX一括export ドライバ。

analysis/eval/run_eval_matrix.py と同じ設計思想:
  - このファイル自体は isaaclab/isaacsim/omni/torch を一切importしない
    （pandas + stdlibのみ）。GPUは scripts/rsl_rl/export_onnx.py 側でのみ使う。
  - 対象は analysis.harness.discover.discover_all_runs() が
    status=="completed" と判定した run のみ（進行中/クラッシュ済みは自動除外）。
  - デフォルトは --dry_run。人間が出力コマンドを確認してから
    --dry_run を外して実行すること（GPUを不用意に長時間占有しないため）。

やること:
  1. 完了済みrunを全て列挙し、(model, seed) ごとに
     scripts/rsl_rl/export_onnx.py の呼び出しコマンドを組み立てる。
     agent / lookahead_horizon / frame_stacking は run_eval_matrix.py の
     AGENT_BY_MODEL / MODEL_ENV_OVERRIDES をそのまま再利用する
     （評価マトリクスと矛盾しない値を使うため、ここで再定義しない）。
  2. --dry_run（デフォルト）: コマンド一覧とサマリだけ表示。何も実行しない。
  3. --dry_run を外すと、1 checkpoint ずつ順番に subprocess 実行
     （GPUは共有リソースなので並列化しない）。
  4. 成功した export について、onnx パッケージ（GPU不要）で
     入力次元をその場で検算し、models/RAL/manifest_fragment.yaml に
     jetson_project の manifest.yaml と同じ形式で書き出す。
     ズレていたら経緯を表示して該当行を書かない（黙って合わせない）。

Usage:
  python analysis/export/export_onnx_matrix.py --dry_run          # まずこれ
  python analysis/export/export_onnx_matrix.py                    # 確認後、実行
  python analysis/export/export_onnx_matrix.py --model A --model E  # 一部モデルだけ
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.harness.discover import discover_all_runs  # noqa: E402
from analysis.eval.run_eval_matrix import AGENT_BY_MODEL, MODEL_ENV_OVERRIDES, TASK_ID  # noqa: E402

STAGING_DIR = REPO_ROOT / "models" / "RAL" / "staging"
FRAGMENT_PATH = REPO_ROOT / "models" / "RAL" / "manifest_fragment.yaml"
MODEL_LABELS = {
    "A": "A: LSTM, lookahead=0.1s",
    "B": "B: LSTM, lookahead=0.5s",
    "C": "C: LSTM, lookahead=1.0s",
    "D": "D: MLP, lookahead=0.5s",
    "E": "E: MLP+framestack(k=5), lookahead=0.5s",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true", default=True,
                   help="コマンドを表示するだけ（デフォルトON）。")
    p.add_argument("--execute", action="store_true", default=False,
                   help="実際に実行する。これを付けない限り何も実行されない。")
    p.add_argument("--model", action="append", default=None,
                   help="A/B/C/D/E のうち対象を絞る（複数指定可）。省略時は全モデル。")
    p.add_argument("--python", type=str, default=sys.executable, help="isaaclab環境のpython実行ファイル")
    p.add_argument("--num_envs", type=int, default=1)
    return p.parse_args()


def build_jobs(models: list[str] | None) -> pd.DataFrame:
    df = discover_all_runs(REPO_ROOT / "logs" / "rsl_rl")
    df = df[df["status"] == "completed"].copy()
    if models:
        df = df[df["model"].isin(models)]
    df = df.sort_values(["model", "seed"]).reset_index(drop=True)
    return df


def build_command(python_exe: str, row, num_envs: int) -> list[str]:
    model = row.model
    overrides = MODEL_ENV_OVERRIDES[model]
    checkpoint = Path(row.run_dir) / f"model_{row.completed_iter}.pt"
    out_name = f"model{model}_seed{row.seed}"
    cmd = [
        python_exe, "scripts/rsl_rl/export_onnx.py",
        "--task", TASK_ID,
        "--agent", AGENT_BY_MODEL[model],
        "--checkpoint", str(checkpoint),
        "--num_envs", str(num_envs),
        "--lookahead_horizon", str(overrides["lookahead_horizon"]),
        "--out_dir", str(STAGING_DIR),
        "--out_name", out_name,
        "--headless",
    ]
    if overrides["use_frame_stacking"]:
        cmd += ["--use_frame_stacking", "--frame_stack_k", str(overrides["frame_stack_k"])]
    return cmd


def expected_obs_dim(model: str) -> int:
    o = MODEL_ENV_OVERRIDES[model]
    base = 10 + int(round(o["lookahead_horizon"] / 0.02))
    return base * (o["frame_stack_k"] if o["use_frame_stacking"] else 1)


def verify_and_fragment(jobs: pd.DataFrame) -> dict:
    """export済みonnxの入力次元を検算し、manifestフラグメントを組み立てる。"""
    try:
        import onnx
    except ImportError:
        print("[verify] onnxパッケージが無いので検算をスキップします（pip install onnx）。")
        return {}

    fragment: dict[str, dict] = {}
    for row in jobs.itertuples():
        model = row.model
        out_name = f"model{model}_seed{row.seed}"
        onnx_path = STAGING_DIR / f"{out_name}.onnx"
        if not onnx_path.exists():
            print(f"[verify] SKIP {out_name}: ファイルが無い（export失敗？）")
            continue
        m = onnx.load(str(onnx_path))
        actual = m.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        expected = expected_obs_dim(model)
        if actual != expected:
            print(f"[verify] NG {out_name}: obs次元 期待{expected} 実際{actual} "
                  f"-> manifest_fragmentに書きません。export_onnx.pyの引数を確認してください。")
            continue
        print(f"[verify] OK  {out_name}: obs={actual}")
        overrides = MODEL_ENV_OVERRIDES[model]
        key = f"{model}_seed{row.seed}"
        fragment[key] = {
            "file": f"RAL/{out_name}.onnx",
            "arch": "lstm" if model in ("A", "B", "C") else "mlp",
            "lookahead_horizon": overrides["lookahead_horizon"],
            "frame_stack": overrides["frame_stack_k"] if overrides["use_frame_stacking"] else 1,
            "label": MODEL_LABELS[model],
            "seed": int(row.seed),
            "checkpoint": str(Path(row.run_dir) / f"model_{row.completed_iter}.pt"),
        }
    return fragment


def main():
    args = parse_args()
    dry_run = not args.execute

    jobs = build_jobs(args.model)
    if jobs.empty:
        print("[export_onnx_matrix] 対象になる completed run がありません。"
              "discover_all_runs() の出力を確認してください。")
        return

    print(f"[export_onnx_matrix] 対象 {len(jobs)} 件 (完了済みrunのみ)")
    print(jobs.groupby("model").size().rename("count").to_string())
    print()

    commands = [build_command(args.python, row, args.num_envs) for row in jobs.itertuples()]

    for row, cmd in zip(jobs.itertuples(), commands):
        print(f"--- {row.model} seed{row.seed} (iter {row.completed_iter}) ---")
        print(" ".join(cmd))
        print()

    if dry_run:
        print("=== --dry_run のため何も実行していません。===")
        print("内容を確認してから `--execute` を付けて再実行してください。")
        return

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for row, cmd in zip(jobs.itertuples(), commands):
        print(f"\n>>> RUN {row.model} seed{row.seed}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"[export_onnx_matrix] FAILED: {row.model} seed{row.seed} (returncode={result.returncode})")
            failed.append((row.model, row.seed))

    print("\n=== export完了。検算とmanifestフラグメント生成 ===")
    fragment = verify_and_fragment(jobs)
    if fragment:
        FRAGMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FRAGMENT_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump({"RAL": fragment}, f, allow_unicode=True, sort_keys=False)
        print(f"[export_onnx_matrix] manifestフラグメントを書き出し: {FRAGMENT_PATH}")
        print("  -> jetson_project/models/manifest.yaml の RAL: セクションにこの内容をマージしてください。")

    if failed:
        print(f"\n[export_onnx_matrix] 失敗 {len(failed)} 件: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
