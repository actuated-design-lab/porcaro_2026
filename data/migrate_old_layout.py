"""旧ディレクトリ配置 → data/ 配下への一回限りの移行スクリプト。

背景
----
2026-10 のディレクトリ整理で、データ系ディレクトリを data/<user>/<学会・目的>/ 配下へ移した。
対応する旧配置は2つ:
  (1) ルート直下（eval_logs*/, eval_assets/, paper_data/, models/, out/）… 2026-10-06 以前
  (2) data/<学会>/（data/ral2026/, data/jfps2026/）… 2026-10-06〜10-07 の一時的な配置
あわせて、学習ログも logs/rsl_rl/ → logs/<user>/rsl_rl/ へ移す（2026-10-07〜、user ごとに分離）。
git が動かすのは「追跡されているファイル」だけなので、.gitignore 対象の実データ
（simulation_log.csv, *.onnx, hardware/ablation/validation の CSV など）は
git pull 後も旧ディレクトリに取り残される。このスクリプトはそれを新しい場所へ移す。

使い方（リポジトリのルートで、ブランチを pull / checkout した後に）
---------------------------------------------------------------
  python data/migrate_old_layout.py            # 何が動くかを表示するだけ（既定）
  python data/migrate_old_layout.py --apply    # 実際に移動する
  python data/migrate_old_layout.py --logs-owner user1 --apply   # 学習ログを user1 のものとして移す

--logs-owner（既定 user0）: このマシンの logs/rsl_rl/ にある学習ログが誰のものか。
  logs/rsl_rl/ に複数人のランが混ざっている場合は、先に手で分けてから実行すること。

挙動
----
- 移動先に同名のファイルが無ければそのまま移動する。
- 同名ファイルが既にあり、中身が同一なら旧側を削除する（git が既に移した分）。
- 同名ファイルが既にあり、中身が違う場合は何もせず「衝突」として報告する。
- 空になった旧ディレクトリは削除する。移しきれなかったものは最後に一覧表示する。
- 何度実行しても安全（既に移したものはスキップされる）。

移行がすべて終わったら、このスクリプトは削除してよい。
"""
from __future__ import annotations

import argparse
import re
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (旧パス, 新パス) — リポジトリルートからの相対パス
MAPPING: list[tuple[str, str]] = [
    # (1) ルート直下の旧配置
    ("eval_assets/midi", "data/common/midi"),
    ("eval_logs", "data/user0/ral2026/eval/main"),
    ("eval_logs_nondr", "data/user0/ral2026/eval/nondr"),
    ("eval_logs_trials5", "data/user0/ral2026/eval/trials5"),
    ("eval_logs_tau_sweep", "data/user0/ral2026/eval/tau_sweep"),
    ("eval_logs_mask_zero", "data/user0/ral2026/eval/mask_zero"),
    ("eval_logs_mask_noise", "data/user0/ral2026/eval/mask_noise"),
    ("eval_logs_mask_shuffle", "data/user0/ral2026/eval/mask_shuffle"),
    ("paper_data", "data/user0/ral2026/paper"),
    ("models/RAL", "data/user0/ral2026/models"),
    ("out/sim", "data/user0/jfps2026/replay"),
    # (2) data/<学会>/ の一時配置
    ("data/ral2026", "data/user0/ral2026"),
    ("data/jfps2026", "data/user0/jfps2026"),
]

# 学習ログ（{owner} は --logs-owner で指定。user ごとに logs/<user>/ へ分ける）
LOGS_MAPPING: list[tuple[str, str]] = [
    ("logs/rsl_rl", "logs/{owner}/rsl_rl"),
    ("logs/rsl_rl_tau_sweep", "logs/{owner}/rsl_rl_tau_sweep"),
    ("logs/experiment_matrix_manifest.json", "logs/{owner}/experiment_matrix_manifest.json"),
]

# 中身が空になったら消してよい親ディレクトリ
PARENTS_TO_PRUNE = ["eval_assets", "models", "out"]

# 旧配置の残骸（中身に意味が無いので消す）
STALE_FILES = ["run_all.pid"]


class Stats:
    def __init__(self) -> None:
        self.moved = 0
        self.dedup = 0
        self.conflicts: list[tuple[Path, Path]] = []


def merge_move(src: Path, dst: Path, apply: bool, stats: Stats) -> None:
    """src を dst へ再帰的にマージ移動する。"""
    if src.is_dir() and not src.is_symlink():
        if not dst.exists():
            print(f"  move  {src.relative_to(REPO_ROOT)}/  ->  {dst.relative_to(REPO_ROOT)}/")
            stats.moved += sum(1 for p in src.rglob("*") if p.is_file())
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            return
        if not dst.is_dir():
            stats.conflicts.append((src, dst))
            return
        for child in sorted(src.iterdir()):
            merge_move(child, dst / child.name, apply, stats)
        if apply and src.exists() and not any(src.iterdir()):
            src.rmdir()
        return

    # ファイル
    if not dst.exists():
        print(f"  move  {src.relative_to(REPO_ROOT)}  ->  {dst.relative_to(REPO_ROOT)}")
        stats.moved += 1
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    elif dst.is_file() and filecmp.cmp(src, dst, shallow=False):
        stats.dedup += 1
        if apply:
            src.unlink()
    else:
        stats.conflicts.append((src, dst))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="実際に移動する（指定しなければ表示のみ）")
    ap.add_argument("--logs-owner", default="user0",
                    help="このマシンの logs/rsl_rl/ の学習ログが誰のものか（既定 user0）")
    args = ap.parse_args()
    if not re.fullmatch(r"user\d+", args.logs_owner):
        print(f"ERROR: --logs-owner は user0, user1, ... の形で指定すること: {args.logs_owner}")
        return 1

    if not (REPO_ROOT / "data").is_dir():
        print("ERROR: data/ が無い。整理後のブランチを pull / checkout してから実行すること。")
        return 1

    mode = "APPLY" if args.apply else "DRY RUN（--apply で実行）"
    print(f"=== 旧配置 -> data/ 移行  [{mode}] ===\n")

    stats = Stats()
    mapping = MAPPING + [(old, new.format(owner=args.logs_owner)) for old, new in LOGS_MAPPING]
    for old, new in mapping:
        src, dst = REPO_ROOT / old, REPO_ROOT / new
        if not src.exists():
            continue
        print(f"[{old}]")
        merge_move(src, dst, args.apply, stats)

    for name in STALE_FILES:
        p = REPO_ROOT / name
        if p.exists():
            print(f"  delete {name}")
            if args.apply:
                p.unlink()

    if args.apply:
        for parent in PARENTS_TO_PRUNE:
            p = REPO_ROOT / parent
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()

    print(f"\n移動: {stats.moved} ファイル / 同一のため旧側を削除: {stats.dedup} ファイル")

    if stats.conflicts:
        print(f"\n[衝突] 移動先に中身の違う同名ファイルがあるため残したもの: {len(stats.conflicts)} 件")
        for s, d in stats.conflicts[:30]:
            print(f"  {s.relative_to(REPO_ROOT)}  vs  {d.relative_to(REPO_ROOT)}")
        if len(stats.conflicts) > 30:
            print(f"  ... ほか {len(stats.conflicts) - 30} 件")

    leftovers = [
        REPO_ROOT / p
        for p in [old for old, _ in mapping] + PARENTS_TO_PRUNE
        if (REPO_ROOT / p).exists()
    ]
    if args.apply and leftovers:
        print("\n[残り] まだ存在する旧ディレクトリ（中身を確認して手で整理すること）:")
        for p in leftovers:
            print(f"  {p.relative_to(REPO_ROOT)}/")

    return 1 if stats.conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
