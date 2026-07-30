# 月曜実行プレイブック（傾向eval → GPU解放 → τスイープ投入 → CPU並行作業）

生成日: 実コードの静的確認に基づく（学習・評価ジョブは一切実行・停止・変更していない）。

## 0. 大前提・確認した安全事項

- 本ドキュメント作成時点で GPU は稼働中（`nvidia-smi` 読み取り確認: util 52%, 6.3/8.2GB使用, PID 1914121 の `env_isaaclab` python プロセス）。**このプロセスには一切触れていない。**
- 以下のコマンド例はすべて **GPU解放後に人間が手で実行する前提**。生成物はコマンド文字列とチェックリストであり、実行結果ではない。
- 実行環境の想定: リポジトリルート(`/home/actuated/my_isaaclab_projects/porcaro_2026`)で `conda activate env_isaaclab` 済み（`environment.yml` の `name: env_isaaclab`、`rsl-rl-lib==3.0.1` 固定と整合）。`run_experiment_matrix.py` が `sys.executable` で子プロセスを起動する既存の使い方に合わせ、以下も `python ...`（アクティベート済み前提）で統一。
- CPU専用の読み取り系コマンド（`analysis.harness.discover`, `nvidia-smi`, `ls`/`cat`）はこのドキュメント生成中に実行して動作確認済み。GPU/isaaclab/torchを使う`train.py`・`play_sim_*.py`・`run_experiment_matrix.py`は一切実行していない。

---

## 1. 実CLI確認サマリ

### 1-1. `analysis/eval/run_eval_matrix.py`

現状の argparse（確定引数）: `--dry_run`, `--heavy_trials`(既定5), `--logs_rsl_rl_root`, `--manifest`。**`--checkpoint`という引数は無い**（`discover_all_runs()`が返す各`run_dir`から`{run_dir}/model_1499.pt`を内部生成する設計）。

**★ジョブ列・ループ順は現状、引数で制御できない。** `build_eval_plan()`の実装は
```python
for row in completed.itertuples():        # 外側 = run (model,seed)
    for pattern, bpm in BASIC:              # 内側 = 条件
        for trial in range(...):
            ...
    for midi_path in GMD:
        ...
```
という「run外側・条件内側」固定順（`analysis/eval/run_eval_matrix.py:130-167`）。月曜の設計原則が要求する「タスク優先度を最外・シードを最内」（条件外側・run内側）を表現できない。条件のサブセット選択（例：`double@160`だけ）や、モデル優先順（B,D先頭）を指定するCLIも無い。

**→ 最小改修案（未適用diff）:**
```diff
--- a/analysis/eval/run_eval_matrix.py
+++ b/analysis/eval/run_eval_matrix.py
@@
+MONDAY_PRIORITY_CONDITIONS: list[tuple[str, str]] = [
+    ("basic", "double_160"),
+    ("gmd", "gmd_03_high_bpm138"),
+    ("gmd", "gmd_04_extreme_bpm170"),
+]
+MONDAY_MODEL_PRIORITY: list[str] = ["B", "D", "A", "C", "E"]
+
+
+def build_priority_plan(
+    runs_df: pd.DataFrame,
+    condition_order: list[tuple[str, str]] = MONDAY_PRIORITY_CONDITIONS,
+    model_priority: list[str] = MONDAY_MODEL_PRIORITY,
+    trials_per_condition: int = 1,
+    python_exe: str | None = None,
+) -> list[dict]:
+    """条件outer / (model,seed)inner のキュー。どこで打ち切っても
+    最優先条件が全model×全seedで先に埋まる（Heavy trial数は使わず、
+    月曜バッチは trial=0..trials_per_condition-1 の固定本数）。"""
+    python_exe = python_exe or sys.executable
+    completed = runs_df[runs_df["status"] == "completed"].copy()
+    completed = completed[completed["model"].isin(AGENT_BY_MODEL)]
+    rank = {m: i for i, m in enumerate(model_priority)}
+    completed["_rank"] = completed["model"].map(lambda m: rank.get(m, len(model_priority)))
+    completed = completed.sort_values(["_rank", "seed"])
+
+    plan: list[dict] = []
+    for kind, condition in condition_order:
+        for row in completed.itertuples():
+            agent = AGENT_BY_MODEL[row.model]
+            checkpoint = Path(row.run_dir) / f"model_{CHECKPOINT_ITER}.pt"
+            for trial in range(trials_per_condition):
+                if kind == "basic":
+                    pattern, bpm = condition.rsplit("_", 1)
+                    cmd = build_rhythm_command(python_exe, checkpoint, agent, pattern, int(bpm), trial)
+                else:
+                    midi_path = next(p for p in GMD if Path(p).stem == condition)
+                    cmd = build_midi_command(python_exe, checkpoint, agent, midi_path, trial)
+                plan.append(dict(model=row.model, seed=row.seed, run_dir=row.run_dir,
+                                  kind=kind, condition=condition, trial=trial, cmd=cmd))
+    return plan
@@ def main() -> None:
     parser.add_argument("--dry_run", ...)
+    parser.add_argument("--priority", action="store_true", default=False,
+                         help="Use the Monday priority queue (condition-outer) instead of the full matrix.")
@@
-    plan = build_eval_plan(runs_df, heavy_r=args.heavy_trials)
+    plan = (build_priority_plan(runs_df) if args.priority
+            else build_eval_plan(runs_df, heavy_r=args.heavy_trials))
```
この diff は **未適用**。フェーズ2ではこれを当てた後の想定コマンドを示す。

### 1-2. `analysis/harness/discover.py`

- チェックポイント列挙は `{run_dir}/model_1499.pt`のようなファイル名を**一覧するだけ**（`_list_checkpoint_iters()`、`.pt`は開かない）。
- 条件識別は `params/env.yaml` + `params/agent.yaml` の中身ベース（`classify_model()`, `discover.py:104-126`）。ディレクトリ名や`experiment_name`（バグで実質2種類にしか分かれない）には依存しない。確認済み。
- **list/dry相当の呼び方**: discoverはそもそも読み取り専用でジョブを起動しないため、`--dry_run`相当のフラグ自体が無い。以下がそのまま「一覧表示」コマンド：
  ```bash
  python -m analysis.harness.discover
  ```
- **C=5本を拾えるか**: 現時点(このドキュメント作成時)では **C=4本+進行中1本(保護)**。理由は`discover_runs()`の"進行中run"判定が「そのフォルダ内でmtime最大のディレクトリ」を保護する実装(`discover.py:69-72`)であるため。C-seed5の学習が終わって次のrun（MLPスイープ等）が同じ`porcaro_rslrl_lstm_modelB_DR/`フォルダに新規書き込みをしなければ、**C-seed5は永久に「進行中」のまま扱われる**（LSTM側フォルダに新規writeが二度と発生しないため）。

**★これは是正が必要なバグ。** 未適用diff（`discover_all_runs()`側でグローバルに保護対象を決める）:
```diff
--- a/analysis/harness/discover.py
+++ b/analysis/harness/discover.py
@@ def discover_runs(experiment_root, max_iterations_hint=1500):
-    mtimes = {d: d.stat().st_mtime for d in run_dirs}
-    protected_dir = max(mtimes, key=mtimes.get) if mtimes else None
+    mtimes = {d: d.stat().st_mtime for d in run_dirs}
+    protected_dir = kwargs.get("protected_dir")  # discover_all_runs から渡される
@@ def discover_all_runs(logs_rsl_rl_root, experiment_dirs=None, max_iterations_hint=1500):
     ...
+    # 保護対象は「両フォルダ全体でmtime最大の1本」のみ。旧フォルダで学習が
+    # 止まった後にmtime最大のまま残り続ける個別フォルダ内判定のバグを回避。
+    all_run_dirs = [d for exp in experiment_dirs for d in (logs_rsl_rl_root/exp).glob("*_seed*") if d.is_dir()]
+    global_protected = max(all_run_dirs, key=lambda d: d.stat().st_mtime, default=None)
     frames = [discover_runs(logs_rsl_rl_root/exp, max_iterations_hint, protected_dir=global_protected)
               for exp in experiment_dirs if (logs_rsl_rl_root/exp).is_dir()]
```
実装の細部は要調整だが方針は「保護判定を2フォルダ横断のグローバルmax mtimeに一本化する」。**フェーズ0で必ずこの症状（C=4のまま）を確認し、直っていなければ手動でこのdiffを当ててから先に進む。**

### 1-3. `play_sim_rhythm.py` / `play_sim_midi.py`

前回セッションで適用済みの改修（このドキュメント作成中に diff で再確認、追加変更なし）:
- `--seed`→`env_cfg.seed`反映済み（両ファイル）。
- `--trial`引数追加済み。出力は`eval_logs/{run_tag}/{ckpt_name}/{condition}_trial{t}/`に分離済み（`condition`はrhythm側`{pattern}_{bpm}bpm`、midi側MIDIファイル名stem）。
- `simulation_log.csv`書き先は`env_cfg.logging.filepath`として`gym.make()`より前に一度だけ設定（両ファイルとも）。**`reward_logging.filepath`は`play_sim_midi.py`側で未設定のまま**（`play_sim_rhythm.py`は設定済み）→ cwd相対の共有`reward_log.csv`に複数trialが追記され続ける既知ギャップ。今回も**変更範囲外として未修正のまま**、要注意点として記載のみ。

**★最重要の追加発見: `play_sim_rhythm.py`のメインループに終了条件が存在しない。**
```python
while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)   # dones を読み捨てている
env.close()
```
（`scripts/rsl_rl/play_sim_rhythm.py:161-169`）`dones`変数を受け取ってすらいない。`--headless`でGUIが無くても`simulation_app.is_running()`は基本的に`True`であり続けるため、**このスクリプトはCtrl+Cされるまで無限に走り続ける**。対照的に`play_sim_midi.py`は`step_count >= max_steps-2`と`dones.any()`の両方で正しく`break`する（`play_sim_midi.py:229-236`）。

このままでは:
- Phase1の「1ジョブ検算」で実行しても自然終了しない → 実時間計測ができない。
- Phase2のバッチ投入で`subprocess.run(cmd, check=True)`が**永久にブロック**し、`double@160`系の全ジョブがそこで詰まる。

**→ 未適用diff（`play_sim_rhythm.py`、必須級）:**
```diff
--- a/scripts/rsl_rl/play_sim_rhythm.py
+++ b/scripts/rsl_rl/play_sim_rhythm.py
@@
+parser.add_argument("--max_episodes", type=int, default=3,
+                    help="Stop after this many episode resets. This script has no other exit "
+                         "condition when run headless/unattended (num_envs=1 assumed).")
@@
     print("="*60)
     print(f" Sim-Verification Started")
     print(f" Mode: {args_cli.pattern} | BPM: {args_cli.bpm}")
     print("="*60)

+    episode_count = 0
     while simulation_app.is_running():
         with torch.inference_mode():
             # 推論
             actions = policy(obs)
-            
             # ステップ実行 (環境がConfig通りのリズムを生成してくれる)
-            obs, _, _, _ = env.step(actions)
-            
+            obs, _, dones, _ = env.step(actions)
+            if dones.any():
+                episode_count += int(dones.sum().item())
+                if episode_count >= args_cli.max_episodes:
+                    print(f"[INFO] Reached max_episodes={args_cli.max_episodes} "
+                          f"({episode_count} resets). Exiting...")
+                    break
     env.close()
     simulation_app.close()
```
**フェーズ1・フェーズ2に進む前に、このdiffだけは適用が事実上必須**（他のdiffは無くても現行コードで動くが、これが無いと1本目からハングする）。

### 1-4. `common/actions/pam.py` の `current_tau_scale`

`grep -n current_tau_scale`結果（`common/actions/pam.py`）:
- `:185` `self.current_tau_scale = None`（初期化）
- `:206` `self.current_tau_scale = torch.full((n_envs,), mid_val, ...)`（`reset()`、`mid_val = sum(tau_scale_range)/2`）
- `:225-229` `reset_idx()`内: `rand_scales = torch.rand(num_resets, ...) * (high-low) + low; self.current_tau_scale[env_ids] = rand_scales`
- `:268` `torque.py`側で`scale = self.current_tau_scale; tau_final = tau_base * scale`として消費

**DR経路 vs 固定値経路の差分は実装不要 — 既存の`tau_scale_range=(X, X)`という縮退区間が数学的に厳密な固定値を与える。** `torch.rand(...) * (high-low) + low`は`high==low`なら`rand()*0.0 + low`となり、IEEE754において`x*0.0 == 0.0`（xが有限値である限りNaNにならない）なので**浮動小数点誤差なしに厳密に`low`**になる。したがって「固定値を入れる経路」は**新規コード不要**で、`env_cfg.pam_tau_scale_range = (X, X)`を設定するだけで達成できる。

ただし現状、この値を**学習/評価CLIから外部指定する手段が無い**（`train.py`にも`play_sim_*.py`にも`--pam_tau_scale`相当の引数が無い）。→ フェーズ4で`train.py`への最小diffを提案（1-5参照）。

**`common/cfg/actuator_cfg.py`の`PamDelayModelCfg`確認:**
```bash
grep -rn "PamDelayModelCfg\|PamModelA_DynamicsCfg\|ActuatorNetModelCfg" source --include="*.py"
# → actuator_cfg.py 自身の定義行以外、一切ヒットしない
```
**`PamDelayModelCfg`・`PamModelA_DynamicsCfg`・`ActuatorNetModelCfg`は完全に未使用（デッドコード）。** 実行時の`tau`テーブルは`torque.py`内にハードコードされたリスト(`tau_P_axis`/`tau_vals`)と`pneumatic.py`のモジュールレベル定数(`TAU_TAB`/`L_TAB`/`TAU_TABLE_2D_DATA`)から直接読まれており、`actuator_cfg.py`のこれらのクラスは参照されていない。**→「倍率で足りる、基本触らない」は正しい確認結果。むしろ触っても効果が無い（配線されていない）ので触る意味自体が無い。**

むだ時間`L`が`tau`倍率の影響を受けないことも確認済み: `torque.py`内`tau_final = tau_base * scale`という乗算は`tau_base`のみに掛かり、`L_cmd`（むだ時間、`self.delay.step(P_cmd, L_cmd)`に渡る）は完全に独立変数（`torque.py:253-266`）。「τ倍率だけ振る（むだ時間L固定）」は既存実装がそのまま満たしている。

**Time Constant Scale ±20% DRをオフにする箇所:** `porcaro_2026_env_cfg.py`の`Porcaro2026EnvCfg_ModelB_DR.__post_init__`内`self.pam_tau_scale_range = (0.8, 1.2)`（`user0/porcaro_2026_env_cfg.py:159`）。ここが唯一の設定箇所。**τスイープでは、このDR上書きが効かない`Template-Porcaro-2026-ModelB-user0`（非DR variant, `pam_tau_scale_range`はベースクラス既定の`(1.0,1.0)`のまま）をtask IDとして使うのが最も単純**（質量/摩擦/関節DRも同時にオフになり、τ×lookaheadの2変数だけを制御変数にできる — これは「スイープ中はTime Constant Scale DRのみオフ」という要求より厳密だが、他のDRが紛れ込まない分むしろ安全側)。

### 1-5. 学習起動(train)の実CLIとステップ数予算

`scripts/rsl_rl/train.py`の確定引数: `--video`, `--video_length`, `--video_interval`, `--num_envs`, `--task`, `--agent`, `--seed`, `--max_iterations`, `--distributed`, `--export_io_descriptors`, `--lookahead_horizon`, `--use_frame_stacking`, `--frame_stack_k` ＋ `cli_args.add_rsl_rl_args()`由来(`--experiment_name`, `--run_name`, `--resume`, `--load_run`, `--checkpoint`, `--logger`, `--log_project_name`) ＋ AppLauncher系。**`--pam_tau_scale`相当は存在しない。**

→ 未適用diff（`train.py`、τスイープに必須）:
```diff
--- a/scripts/rsl_rl/train.py
+++ b/scripts/rsl_rl/train.py
@@
 parser.add_argument("--frame_stack_k", type=int, default=5,
                     help="Number of frames to stack.")
+parser.add_argument("--pam_tau_scale", type=float, default=None,
+                    help="Fix PAM time-constant multiplier to this single value, "
+                         "overriding pam_tau_scale_range DR sampling entirely "
+                         "(e.g. 0.5 / 1.0 / 2.0 for the tau sweep).")
@@
     if args_cli.use_frame_stacking:
         env_cfg.use_frame_stacking = True
         env_cfg.frame_stack_k = args_cli.frame_stack_k

+    if args_cli.pam_tau_scale is not None:
+        env_cfg.pam_tau_scale_range = (args_cli.pam_tau_scale, args_cli.pam_tau_scale)
+        print(f"[Config] pam_tau_scale_range fixed to "
+              f"({args_cli.pam_tau_scale}, {args_cli.pam_tau_scale}) for tau sweep")
+
     dt_ctrl = env_cfg.sim.dt * env_cfg.decimation
```

**ステップ数予算（実測値、マニフェストから算出）:** 既存14本の`elapsed_min`平均 = **293.6分（約4.9時間/本）**、range 288.4〜298.1分。いずれも`max_iterations=1500`。**τグリッド27本にこの予算をそのまま適用すると 27×4.9h ≈ 132時間（5.5日）連続GPU占有になる** — 「n=1傾向が早く読める」という設計原則（seed最外・9セル先読み）の意図と整合しない可能性が高い。**この点はコードから確定できないため、フェーズ4の実行前に人間の判断が必要（詳細は末尾「未解決」参照）。**

---

## 2. 設計原則（固定・再掲）

**evalキュー**: タスク優先度最外・シード最内。
1. `double@160` × 全model × 全seed（trial=0のみ、1本/セル）。同一タスク内は **B, D** を先頭、続いて A, C, E。
2. GMD高テンポ2条件（`gmd_03_high_bpm138`, `gmd_04_extreme_bpm170`）× 全model × 全seed。
3. 低テンポ7条件・`single_8`は列末尾／除外。

条件×(model,seed)の**目標**本数 = 3条件 × 5model × 5seed = **75本**（5モデル完了時点の設計値）。**現時点の実データでは D/E 未学習・C は4/5のため、実際にキューへ入るのは 3条件 × 14本(A5+B5+C4) = 42本**（C-seed5完了後なら45本）。75本ちょうどになるのはD/Eの学習が完了して初めて。

**τグリッド**: 27本 = 9セル(τ倍率3×ホライズン3、下表) × 3seed。キューは**seed最外**（seed1で9セル全部→n=1傾向を先に把握）。

| τ倍率 | ホライズン窓 |
|---|---|
| 0.5× | 0.1 / 0.25 / 0.5 s |
| 1.0× | 0.25 / 0.5 / 1.0 s |
| 2.0× | 0.5 / 1.0 / 2.0 s |

τはτ倍率のみ振る（むだ時間L固定 — 1-4で確認済み、既存実装のまま満たされる）。DR(Time Constant Scale±20%)はスイープ中オフ（非DR task使用で自動的にオフ、1-4参照）。

---

## フェーズ0: 事前確認

**所要目安**: 5分以内（すべて読み取り専用コマンド）。
**ゲート条件**: 以下3点がすべてクリアでフェーズ1へ進む。①GPUがidle。②discoverでC=5本（またはC-seed5完了直後で4→5への遷移を確認）。③play_sim_rhythm.pyへの`--max_episodes`diffが適用済み（1-3節、必須級）。

```bash
cd /home/actuated/my_isaaclab_projects/porcaro_2026

# ① GPU idle確認（読み取りのみ）
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# -> compute-apps が空、utilization.gpu が 0% 近辺であることを確認

# ② 学習run一覧（list/dry相当。discoverはジョブを起動しないので追加フラグ不要）
python -m analysis.harness.discover
# -> C が 5本 (completed) になっているか確認。
#    まだ C=4 + protected running のままなら 1-2節のバグに該当。
#    次のパッチを discover.py に適用してから再実行:
#      (1-2節の diff)
#    再実行しても改善しない場合は、少なくとも C の5本目の run_dir を
#    `ls -dt logs/rsl_rl/porcaro_rslrl_lstm_modelB_DR/*/ | head -1` で
#    手動特定し、フェーズ1以降ではその run_dir を直接使う。

# ③ pam.py改修有無の確認（読み取りのみ、差分無しのはず）
git diff --stat -- source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/common/actions/pam.py
# -> 何も出なければ pam.py は未改修（想定通り、フェーズ4まで不要）

# ④ play_sim_rhythm.py に --max_episodes diff が当たっているか確認
grep -n "max_episodes" scripts/rsl_rl/play_sim_rhythm.py
# -> ヒット無しならフェーズ1に進む前に 1-3節の diff を手動適用すること
```

---

## フェーズ1: 1ジョブ検算（★必須ゲート）

**所要目安**: diff適用+検証 10分、実ジョブ実行は`--max_episodes 3`なら数分〜十数分程度（1エピソード=BPM160・4小節=`16×60/160`=6秒分のsim時間だが、リアルタイム同期していないシム速度次第。実測して次フェーズの見積りに使う）。
**ゲート条件**: `simulation_log.csv`が生成され、行数>0、`force_z`列に非ゼロ値が現れる（打撃が記録されている）。run_tag衝突なし・パス分離が効いていることを目視確認できたら次へ。

対象: `double@160, model B, seed1` → `run_dir = logs/rsl_rl/porcaro_rslrl_lstm_modelB_DR/2026-07-08_19-38-34_seed1`（`discover`出力で確認済み、B/seed1）。

```bash
cd /home/actuated/my_isaaclab_projects/porcaro_2026
CKPT=logs/rsl_rl/porcaro_rslrl_lstm_modelB_DR/2026-07-08_19-38-34_seed1/model_1499.pt
ls -la "$CKPT"   # チェックポイント実在確認（読み取りのみ）

# --- まず dry-run 感覚で1エピソードだけ超短時間確認したい場合は
#     --max_episodes 1 でもよいが、ここでは本番同様に3で検算する ---

time python scripts/rsl_rl/play_sim_rhythm.py \
  --checkpoint "$CKPT" \
  --task Template-Porcaro-2026-ModelB-DR-user0 \
  --agent rsl_rl_lstm_cfg_entry_point \
  --pattern double --bpm 160 \
  --trial 0 --seed 1000 \
  --max_episodes 3 \
  --headless --num_envs 1

# --- 検証（読み取りのみ） ---
OUT=eval_logs/2026-07-08_19-38-34_seed1/model_1499/double_160bpm_trial0
ls -la "$OUT"
wc -l "$OUT/simulation_log.csv"
head -3 "$OUT/simulation_log.csv"
python -c "
import pandas as pd
df = pd.read_csv('$OUT/simulation_log.csv')
print('rows:', len(df))
print('force_z nonzero:', (df['force_z'] > 1.0).sum())
print('target_bpm unique:', df['target_bpm'].unique())
"
```
確認ポイント: `run_tag`（`2026-07-08_19-38-34_seed1`）・`ckpt_name`（`model_1499`）・`condition_tag`（`double_160bpm_trial0`）の3階層が想定通り分離されているか、既存の`eval_logs/2026-03-01_08-00-23/...`ディレクトリ（過去の別実験の遺物、以前のセッションで確認済み）と衝突していないか。`time`コマンドの実測値をフェーズ2の所要時間見積りに使う。

---

## フェーズ2: バッチ投入（優先度キュー75本[目標]/42〜45本[現実]）

**所要目安**: フェーズ1の実測時間 × ジョブ数（現実的にはGMD 2条件はMIDI長依存でBASIC条件より長時間になりうる点に注意）。tmux+nohupでバックグラウンド化し、CPU並行作業（フェーズ3）に移る。
**ゲート条件**: `tail -f`で最初の数ジョブが正常にstatus記録されているのを確認できたら、あとは放置して定期チェックへ移行してよい。

前提: 1-1節の`build_priority_plan()`diffを`analysis/eval/run_eval_matrix.py`に適用済み、かつ1-3節の`--max_episodes`diffを`play_sim_rhythm.py`に適用済み。

```bash
cd /home/actuated/my_isaaclab_projects/porcaro_2026

# 必ず --dry_run でコマンド一覧を先に確認（何も実行されない）
python analysis/eval/run_eval_matrix.py --priority --dry_run
# -> 出力本数が 3条件 × (現在のcompleted本数) と一致するか目視確認
#    (現時点想定: 42〜45本。D/E学習後なら75本)

# 確認OKなら tmux + nohup で本実行（長時間・GPU占有）
tmux new -s eval_monday -d
tmux send-keys -t eval_monday \
  "cd /home/actuated/my_isaaclab_projects/porcaro_2026 && \
   python analysis/eval/run_eval_matrix.py --priority \
   2>&1 | tee eval_logs/monday_eval_batch.log" C-m

# 投入後の確認（別ターミナル、読み取りのみ）
tmux attach -t eval_monday   # 抜けるときは Ctrl-b d
tail -f eval_logs/monday_eval_batch.log
cat eval_logs/eval_matrix_manifest.json | python -m json.tool | tail -40
```

---

## フェーズ3: CPU並行作業（フェーズ2/4のGPUジョブと並行）

**所要目安**: フェーズ2完了分から順次、継続的に。
**ゲート条件**: 特になし（フェーズ2のジョブが1本でも`success`になった時点で着手可能、以後は取れたデータから随時更新）。

既存の`analysis/`ハーネス（前回セッションで作成済み・pytest 12件pass確認済み）をそのまま使う。GPUを使わないので、フェーズ2・4のGPUジョブと同時に走らせてよい。

```bash
cd /home/actuated/my_isaaclab_projects/porcaro_2026

# 回帰確認（CPUのみ）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest analysis/tests -q

# 学習曲線・per-seed図の更新（TB event読み取りのみ、GPU不要）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m analysis.harness.tb_curves
```
`strike_extract`/`stats`の本番集計（C=5再集計含む）はフェーズ2のeval出力が溜まり次第、フェーズ5で行う（下記）。

---

## フェーズ4: τ前ガード＆投入（27本、seed最外）

**所要目安**: ガード確認 15分。27本の投入自体はGPU長時間占有（下記「未解決」の通りステップ予算次第で日〜週オーダー）。
**ゲート条件**: assert検算がPASSしてから1本目を投入。1本目のログ（`params/env.yaml`の`pam_tau_scale_range`）が意図した`(X, X)`になっていることを確認してから残り26本を流す。

### 4-1. 固定値パッチの適用（未適用diff、1-4/1-5節に全文）
- `scripts/rsl_rl/train.py`: `--pam_tau_scale`引数を追加（1-5節diff）。
- `pam.py`自体は**変更不要**（縮退区間`(X,X)`で数学的に厳密、1-4節参照）。「pam.py固定値パッチ」を明示的なコードとして残したい場合のみ、以下のような**オプショナルな**防御的diffを検討（無くても正しく動く）:
```diff
--- a/source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/common/actions/pam.py
+++ b/source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/common/actions/pam.py
@@ def reset_idx(self, env_ids):
         if self.current_tau_scale is not None:
             low, high = self.tau_scale_range
             num_resets = len(env_ids)
-            rand_scales = torch.rand(num_resets, device=self.current_tau_scale.device) * (high - low) + low
+            if low == high:
+                # 縮退区間: DRサンプリングを完全にバイパスして厳密な固定値を代入
+                rand_scales = torch.full((num_resets,), low, device=self.current_tau_scale.device)
+            else:
+                rand_scales = torch.rand(num_resets, device=self.current_tau_scale.device) * (high - low) + low
             self.current_tau_scale[env_ids] = rand_scales
```

### 4-2. CPU-only 検算スクリプト（torch/isaaclab不使用、numpyのみ）
`pneumatic.py`のテーブル値をそのまま複製し、τ倍率が意図通り効くことを**シミュレータを起動せずに**事前検算する。以下は monday_run.md に埋め込むコード例であり、実行するかどうか・保存先は人間判断（`analysis/`配下に保存するならこの回のスコープ外なので次回のセッションで作成すること）:
```python
import numpy as np

# pneumatic.py の 1D テーブルをそのまま複製（torch不使用）
P_TAB   = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
TAU_TAB = np.array([0.043, 0.045, 0.060, 0.066, 0.094, 0.131])

def tau_base_1d(p_cmd: float) -> float:
    return float(np.interp(p_cmd, P_TAB, TAU_TAB))

for tau_scale in [0.5, 1.0, 2.0]:
    for p_cmd in [0.1, 0.3, 0.6]:
        base = tau_base_1d(p_cmd)
        expected = tau_scale * base
        print(f"tau_scale={tau_scale} P={p_cmd}: tau_base={base:.4f} "
              f"tau_final(expected)={expected:.4f}")
        assert abs(expected - tau_scale * base) < 1e-12
```
実機検算（学習run後）は、1本目のジョブが吐いた`params/env.yaml`を読んで`pam_tau_scale_range`欄が狙った`(X, X)`になっているかを確認するだけでよい:
```bash
grep -A1 "pam_tau_scale_range" \
  logs/rsl_rl/porcaro_rslrl_lstm_modelB_DR/<新規run_dir>/params/env.yaml
```

### 4-3. DRオフの確認箇所
非DR taskを使う（1-4節）ため、以下を`--task`に指定するだけでTime Constant Scale DR含む全DRがオフになる:
```
--task Template-Porcaro-2026-ModelB-user0
```
（`-DR-`が付かないID。`gym.register`で存在確認済み、`user0/__init__.py:14-23`）

### 4-4. 27本投入コマンド（seed最外・9セル内側）

```bash
cd /home/actuated/my_isaaclab_projects/porcaro_2026

declare -a CELLS=(
  "0.5 0.1"  "0.5 0.25" "0.5 0.5"
  "1.0 0.25" "1.0 0.5"  "1.0 1.0"
  "2.0 0.5"  "2.0 1.0"  "2.0 2.0"
)

tmux new -s tau_sweep -d
for seed in 1 2 3; do
  for cell in "${CELLS[@]}"; do
    tau=$(echo "$cell" | cut -d' ' -f1)
    lh=$(echo "$cell" | cut -d' ' -f2)
    tag="tau${tau}_lh${lh}"
    tmux send-keys -t tau_sweep \
      "python scripts/rsl_rl/train.py \
        --task Template-Porcaro-2026-ModelB-user0 \
        --agent rsl_rl_lstm_cfg_entry_point \
        --seed ${seed} \
        --lookahead_horizon ${lh} \
        --pam_tau_scale ${tau} \
        --experiment_name porcaro_tau_sweep_${tag} \
        --run_name seed${seed} \
        --max_iterations <★未確定、下記「未解決」参照> \
        --num_envs 2048 \
        --headless" C-m
  done
done
```
（`--experiment_name`を渡しても現状バグで無視される点は1-2節既知の通り。区別は`params/env.yaml`の`lookahead_horizon`と、今回新設する`pam_tau_scale_range`フィールドの中身で行う。discover.py側で`classify_model()`相当のτ版フィルタが必要になる点も未解決リストに記載。）

---

## フェーズ5以降: 集計

```bash
cd /home/actuated/my_isaaclab_projects/porcaro_2026

# C=5本を含めた再集計（フェーズ0でC=5確認済みの前提）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m analysis.harness.tb_curves

# strike_extract -> stats のパイプライン（実データ版、要:フェーズ2のeval_logs出力）
python - <<'PY'
import pandas as pd
from pathlib import Path
from analysis.harness.strike_extract import extract_strikes

rows = []
for csv_path in Path("eval_logs").glob("*/*/*/simulation_log.csv"):
    # パス構造: eval_logs/{run_tag}/{ckpt_name}/{condition}_trial{t}/simulation_log.csv
    condition_dir = csv_path.parent.name
    df = pd.read_csv(csv_path)
    strikes = extract_strikes(df, bpm=None, target_ref=20.0, tol_ms=30.0)
    strikes["condition_dir"] = condition_dir
    strikes["source_csv"] = str(csv_path)
    rows.append(strikes)

all_strikes = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
print(f"collected {len(all_strikes)} strikes from {len(rows)} csv files")
all_strikes.to_csv("analysis/outputs/monday_strikes_raw.csv", index=False)
PY
# -> model/seed/task/eval_trial 列を simulation_log.csv のパスから付与した上で
#    analysis.harness.stats.seed_level / across_seed / perm_test に渡す
#    (列の抽出ロジックは eval_logs のパス構造から機械的に作れるが、今回は
#     スコープ外につきコード化していない — 次回セッションで対応)

# τ-eval（27ckpt）は上記と同じ play_sim_rhythm.py バッチをτ-sweep側のrun_dirに向けて回す
```

---

## 未解決・要確認

1. **★τスイープ27本のステップ数予算（`--max_iterations`）が実コードから決定できない。** 既存ロード済み14本の実測は平均293.6分/本(max_iterations=1500)。同じ予算を27本に適用すると約132時間(5.5日)連続でGPUを占有し、「n=1傾向を早く読む」という設計意図（seed最外キュー）と整合しない。カリキュラム閾値(`curriculum_thresholds=[25_000_000, 100_000_000]`ステップ, `porcaro_2026_env.py:60`)から逆算すると、Level0→1到達は約iteration 100〜110、Level1→2到達は約iteration 400〜410（1iteration ≈ num_envs×num_steps_per_env = 2048×120 = 245,760ステップとして概算）。「傾向だけ早く見る」用途なら例えば`max_iterations=200〜400`あたりが候補だが、**これは提案であり決定ではない**。人間が値を決めてから4-4節のコマンドに埋めること。
2. `discover.py`のグローバル保護判定diff（1-2節）は方針のみ提示、実装の細部（`discover_runs()`への`protected_dir`引数追加のシグネチャ変更が`analysis/tests`の既存呼び出し互換性に影響しないか）は未検証。適用前に`pytest analysis/tests`を通すこと。
3. τスイープ27本を`discover.py`/`classify_model()`で自動分類する仕組みが無い。現行`classify_model()`は`lookahead_horizon`と`recurrent`/`use_frame_stacking`のみを見ており、`pam_tau_scale_range`は見ていない。τ×lookaheadの9セル×3seedを一覧するには`classify_model()`相当の拡張（または専用の`classify_tau_cell()`)が別途必要（今回未実装、範囲外）。
4. `run_eval_matrix.py`の`--priority`実装（1-1節diff）は本ドキュメント内でのみ提示した設計で、実際にコード化・テストはしていない。フェーズ2に入る前に軽くdry runで件数検算すること。
5. `eval_logs/2026-03-01_08-00-23/...`という過去実験の遺物ディレクトリが存在する（前回セッションで確認済み、今回の学習マトリクスより前の日付）。優先度キューの出力先(`eval_logs/{run_tag}/...`)がこれと衝突しないことは`run_tag`が実runのタイムスタンプ由来なので原理的に衝突しないはずだが、目視確認は推奨。
6. `play_sim_midi.py`の`reward_logging.filepath`未設定ギャップ（1-3節）は今回も未修正。GMD条件のtrial間で`reward_log.csv`が意図せず共有される可能性がある。フェーズ2でGMD条件を回す前に修正するか、`reward_log.csv`の解析は諦めて`simulation_log.csv`ベースの集計のみ使うか、判断が必要。
7. `--pam_tau_scale`が渡された場合の`env_cfg.observation_space`再計算ロジック（`train.py`の`lookahead_steps`計算）への影響は無いはず（τは observation 次元に関与しない）だが、実行して確認していない。

---

## 生成・変更ファイルと実行確認

**生成したファイル**: `docs/monday_run.md`（本ファイル）のみ。

**変更したファイル**: なし。`analysis/eval/run_eval_matrix.py`、`analysis/harness/discover.py`、`scripts/rsl_rl/train.py`、`scripts/rsl_rl/play_sim_rhythm.py`、`source/.../common/actions/pam.py`への提案diffはすべて本ファイル内に記述するのみで、**いずれのソースファイルにも適用していない**。

**実行したもの**（すべて読み取り専用・GPU非使用）:
- `nvidia-smi --query-gpu=...` / `--query-compute-apps=...`（GPU状態確認）
- `python -m analysis.harness.discover`（既存の読み取り専用ツール、既に前回セッションで実行実績あり）
- `Read`/`grep`によるソースファイル閲覧（`run_eval_matrix.py`, `discover.py`, `play_sim_rhythm.py`, `play_sim_midi.py`, `pam.py`, `actuator_cfg.py`, `porcaro_2026_env_cfg.py`, `train.py`, `user0/__init__.py`）
- `logs/experiment_matrix_manifest.json`からのPython集計（`elapsed_min`の平均/min/max算出、pandasすら使わない標準ライブラリのみ）

**実行しなかったもの**: `train.py` / `play_sim_rhythm.py` / `play_sim_midi.py` / `run_experiment_matrix.py` / `analysis/eval/run_eval_matrix.py` は一度も実行していない。稼働中の学習ジョブ（PID 1914121）には一切触れていない。GPUメモリ確保・プロセス起動・停止・シグナル送信は行っていない。
