# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
export_onnx.py — 1チェックポイントを ONNX(+JIT) に export する（推論ループなし）。

scripts/rsl_rl/play.py の export ロジック（export_policy_as_jit/onnx、
normalizerの自動抽出）と、scripts/rsl_rl/play_sim_rhythm.py の
--lookahead_horizon/--use_frame_stacking/--frame_stack_k オーバーライド
（train.py:178-199と同型。A/C/Eの学習時パラメータをobservation_space計算前に
反映しないと OnPolicyRunner.load() で観測次元不一致エラーになる）を
組み合わせたもの。

play.py 単体だと A/C/E のcheckpointは読めない（デフォルトenv_cfgは
lookahead=0.5・frame_stacking無し=35次元固定のため）。このスクリプトは
モデルごとに正しいenv_cfgへオーバーライドしてから読み込む。

シミュレーションは１ステップも回さない（env構築→checkpoint読込→export→終了）。
GPU占有時間を最小化するため。

Usage:
  python scripts/rsl_rl/export_onnx.py \
      --task Template-Porcaro-2026-ModelB-DR-user0 \
      --agent rsl_rl_lstm_cfg_entry_point \
      --checkpoint=logs/rsl_rl/porcaro_rslrl_lstm_modelB_DR/2026-02-13_16-15-01_seed1000/model_1499.pt \
      --lookahead_horizon 0.1 \
      --out_dir models/RAL/staging --out_name modelA_seed1000

  # E（frame stacking）の場合
  python scripts/rsl_rl/export_onnx.py \
      --task Template-Porcaro-2026-ModelB-DR-user0 \
      --agent rsl_rl_mlp_cfg_entry_point \
      --checkpoint=logs/rsl_rl/.../model_1499.pt \
      --lookahead_horizon 0.5 --use_frame_stacking --frame_stack_k 5 \
      --out_dir models/RAL/staging --out_name modelE_seed1000

通常は手で呼ばず、analysis/export/export_onnx_matrix.py から
（--dry_runでコマンドを確認した上で）呼び出すことを想定。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export a single RSL-RL checkpoint to ONNX (no sim rollout).")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (env構築に必要。1で十分)。")
parser.add_argument("--seed", type=int, default=None, help="Env seed (export結果には影響しない)。")

# train.py:178-199 / play_sim_rhythm.py と同型のオーバーライド
parser.add_argument("--lookahead_horizon", type=float, required=True,
                    help="このcheckpointの学習時lookahead_horizon [s]。必須(黙って0.5扱いにしない)。")
parser.add_argument("--use_frame_stacking", action="store_true", default=False,
                    help="frame stacking(モデルE)を有効化。")
parser.add_argument("--frame_stack_k", type=int, default=1,
                    help="frame stack数。use_frame_stackingと併用。")

parser.add_argument("--out_dir", type=str, required=True, help="出力ディレクトリ")
parser.add_argument("--out_name", type=str, required=True,
                    help="出力ファイル名の拡張子抜き部分。例: modelA_seed1000")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import porcaro_2026.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    resume_path = retrieve_file_path(args_cli.checkpoint)

    # --- ★ ここが play.py に無くて play_sim_rhythm.py にある部分 ---
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env_cfg.lookahead_horizon = args_cli.lookahead_horizon
    env_cfg.use_frame_stacking = bool(args_cli.use_frame_stacking)
    env_cfg.frame_stack_k = args_cli.frame_stack_k if args_cli.use_frame_stacking else 1

    dt_ctrl = env_cfg.sim.dt * env_cfg.decimation
    lookahead_steps = int(round(env_cfg.lookahead_horizon / dt_ctrl))
    base_obs_dim = 10 + lookahead_steps
    env_cfg.observation_space = (
        base_obs_dim * env_cfg.frame_stack_k if env_cfg.use_frame_stacking else base_obs_dim
    )
    print(f"[Export] checkpoint={resume_path}")
    print(f"[Export] lookahead={env_cfg.lookahead_horizon}s ({lookahead_steps} steps) "
          f"frame_stacking={env_cfg.use_frame_stacking} k={env_cfg.frame_stack_k} "
          f"-> observation_space={env_cfg.observation_space}")
    # ------------------------------------------------------------------

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy_nn = None
    try:
        policy_nn = runner.alg.policy               # rsl_rl >= 2.3
    except AttributeError:
        policy_nn = runner.alg.actor_critic          # rsl_rl <= 2.2

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
        print("[Export][WARN] normalizerが見つかりません。ONNXは無正規化で出力されます。"
              "models/RAL/README.md の要件(正規化を焼き込む)を満たさない可能性があるので必ず確認すること。")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=args_cli.out_dir,
                         filename=f"{args_cli.out_name}.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=args_cli.out_dir,
                          filename=f"{args_cli.out_name}.onnx")
    print(f"[Export] -> {os.path.join(args_cli.out_dir, args_cli.out_name)}.onnx")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
