"""
Script to play a checkpoint with SPECIFIC rhythm injection (Sim Verification).
Target: Isaac Lab / RSL-RL
Matches logic with: run_deploy_v4.py

Usage:
  python scripts/rsl_rl/play_sim_rhythm.py --load_run [RunName] --pattern double --bpm 120
"""

import argparse
import sys
import os
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# local imports
import cli_args
import obs_mask

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Play RL agent with Controlled Rhythm Input.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=400, help="Length of video.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Porcaro-Direct-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained ckpt.")

# ★ デプロイスクリプトと共通の引数
parser.add_argument("--bpm", type=float, default=60.0, help="Target BPM (e.g. 60, 120).")
parser.add_argument("--pattern", type=str, default="single_4",
                    choices=["single_4", "single_8", "double", "paradiddle", "upbeat", "clave"],
                    help="Rhythm pattern to test.")
parser.add_argument("--trial", type=int, default=0,
                    help="Trial index for this (pattern, bpm) condition; used to separate output CSVs "
                         "and to derive a reproducible per-trial seed.")
parser.add_argument("--max_episodes", type=int, default=3,
                    help="Stop after this many episode resets. This script has no other exit "
                         "condition when run headless/unattended (num_envs=1 assumed).")
parser.add_argument("--lookahead_horizon", type=float, default=None,
                    help="Override lookahead horizon (e.g. 0.1, 0.5, 1.0). Must match the "
                         "checkpoint's training-time value or policy loading will fail with an "
                         "observation-space size mismatch.")
parser.add_argument("--use_frame_stacking", action="store_true", default=False,
                    help="Enable frame-stacking (finite history) observation.")
parser.add_argument("--frame_stack_k", type=int, default=5,
                    help="Number of frames to stack.")
parser.add_argument("--pam_tau_scale", type=float, default=None,
                    help="Fix PAM time-constant multiplier to this single value, overriding "
                         "pam_tau_scale_range DR sampling entirely. Must match the checkpoint's "
                         "training-time value for the tau sweep (e.g. 0.5 / 1.0 / 2.0).")
parser.add_argument("--eval_logs_root", type=str, default="eval_logs",
                    help="Root directory for eval_logs/{run_tag}/{ckpt}/{condition}_trial{t}/ output "
                         "(override to keep tau-sweep/non-DR/etc. eval runs from mixing with the "
                         "main experiment matrix's eval_logs/).")
parser.add_argument("--mask_mode", type=str, default="none", choices=list(obs_mask.MASK_MODES),
                    help="Far-future lookahead ablation: overwrite the [--mask_lo_s, --mask_hi_s) "
                         "window of the lookahead observation buffer with zeros/uniform noise/a "
                         "random within-window shuffle, applied every step after env.reset()/step().")
parser.add_argument("--mask_lo_s", type=float, default=0.5,
                    help="Start of the masked lookahead time window (seconds).")
parser.add_argument("--mask_hi_s", type=float, default=1.0,
                    help="End of the masked lookahead time window (seconds).")

# RSL-RL args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after Sim Launch ---
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import porcaro_2026.tasks

# --- Main Logic ---
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # 1. パス解決
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)

    # 2. 環境設定のオーバーライド (ここが重要)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    # --seed は元々 argparse にあるだけで env_cfg に渡っていなかったため明示的に反映する
    # (未指定なら env_cfg.seed のデフォルト値のまま = Isaac Lab 側は非決定的に動く)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # train.py:178-199 と同型のオーバーライド。checkpointが学習された
    # lookahead_horizon/use_frame_stacking/frame_stack_kと一致させないと、
    # OnPolicyRunner.load()時にobservation_space不一致でエラーになる。
    if args_cli.lookahead_horizon is not None:
        env_cfg.lookahead_horizon = args_cli.lookahead_horizon
        print(f"[Config] lookahead_horizon overridden to {args_cli.lookahead_horizon}")

    if args_cli.use_frame_stacking:
        env_cfg.use_frame_stacking = True
        env_cfg.frame_stack_k = args_cli.frame_stack_k

    if args_cli.pam_tau_scale is not None:
        env_cfg.pam_tau_scale_range = (args_cli.pam_tau_scale, args_cli.pam_tau_scale)
        print(f"[Config] pam_tau_scale_range fixed to "
              f"({args_cli.pam_tau_scale}, {args_cli.pam_tau_scale}) for tau sweep eval")

    dt_ctrl = env_cfg.sim.dt * env_cfg.decimation
    lookahead_steps = int(env_cfg.lookahead_horizon / dt_ctrl)
    base_obs_dim = 10 + lookahead_steps
    env_cfg.observation_space = (
        base_obs_dim * env_cfg.frame_stack_k if env_cfg.use_frame_stacking else base_obs_dim
    )
    print(f"[Config Override] lookahead={env_cfg.lookahead_horizon} "
        f"frame_stacking={env_cfg.use_frame_stacking} k={env_cfg.frame_stack_k} "
        f"-> observation_space={env_cfg.observation_space}")

    mask_lo_idx, mask_hi_idx = obs_mask.compute_mask_range(
        dt_ctrl, lookahead_steps, args_cli.mask_lo_s, args_cli.mask_hi_s
    )
    if args_cli.mask_mode != "none":
        print(f"[Mask] mode={args_cli.mask_mode} window=[{args_cli.mask_lo_s}s, {args_cli.mask_hi_s}s) "
              f"-> obs columns [{mask_lo_idx}, {mask_hi_idx}) of {env_cfg.observation_space}")
        if env_cfg.use_frame_stacking:
            raise ValueError(
                "--mask_mode is only implemented for the non-frame-stacked observation layout "
                "(obs_mask.compute_mask_range assumes obs_single, not the frame-stacking reshape)."
            )

    # 条件タグを作る（チェックポイント名＋pattern＋bpm＋trialで一意にする）
    ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]   # 例: model_1499
    run_tag = os.path.basename(os.path.dirname(resume_path))          # 例: 2026-03-01_08-00-23
    condition_tag = f"{args_cli.pattern}_{int(args_cli.bpm)}bpm_trial{args_cli.trial}"

    eval_out_dir = os.path.join(args_cli.eval_logs_root, run_tag, ckpt_name, condition_tag)
    os.makedirs(eval_out_dir, exist_ok=True)

    if hasattr(env_cfg, "logging"):
        env_cfg.logging.enabled = True
        env_cfg.logging.filepath = os.path.join(eval_out_dir, "simulation_log.csv")
    if hasattr(env_cfg, "reward_logging"):
        env_cfg.reward_logging.enabled = True
        env_cfg.reward_logging.filepath = os.path.join(eval_out_dir, "reward_log.csv")

    # ★ RhythmGeneratorを「テストモード」にする設定
    # ※ Porcaro2026EnvCfg にこれらのフィールドがある前提ですが、
    #    もしなければ後述の `env.unwrapped` で直接注入します。
    if hasattr(env_cfg, "use_simple_rhythm"):
        env_cfg.use_simple_rhythm = True
        env_cfg.simple_rhythm_mode = args_cli.pattern
        env_cfg.simple_rhythm_bpm = args_cli.bpm
        print(f"[Config] Config Overridden: Pattern={args_cli.pattern}, BPM={args_cli.bpm}")

    # 3. 環境構築
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Video録画設定
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_sim_rhythm"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 4. エージェントロード
    print(f"[INFO]: Loading model from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 5. 直接注入 (念の為の二重対策)
    # Config経由で設定されなかった場合、直接内部のRhythmGeneratorを叩く
    try:
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, "rhythm_generator"):
            print("[INFO] Setting RhythmGenerator Test Mode directly...")
            unwrapped_env.rhythm_generator.set_test_mode(
                enabled=True, 
                bpm=args_cli.bpm, 
                pattern=args_cli.pattern
            )
            # 全環境のリズムをリセットして反映
            unwrapped_env.rhythm_generator.reset(
                torch.arange(unwrapped_env.num_envs, device=unwrapped_env.device)
            )
    except Exception as e:
        print(f"[WARNING] Could not set rhythm generator directly: {e}")

    # --- Simulation Loop ---
    obs, _ = env.reset()

    # ★安全確認: pam_tau_scale が current_tau_scale (pam.py::PAMChannel) に
    # 実際に反映されているかをここで検証する。reset()/reset_idx()がこれを
    # 設定するので、env.reset()直後でないと値がNoneのまま。
    if args_cli.pam_tau_scale is not None:
        raw_env = env.unwrapped
        ctrl = raw_env.action_controller
        expected = args_cli.pam_tau_scale
        for ch_name in ("ch_DF", "ch_F", "ch_G"):
            ch = getattr(ctrl, ch_name)
            actual = ch.current_tau_scale
            assert actual is not None, f"{ch_name}.current_tau_scale is None after env.reset()"
            assert torch.allclose(actual, torch.full_like(actual, expected)), (
                f"[Tau Safety Check] {ch_name}.current_tau_scale={actual.tolist()} != expected {expected} "
                f"-- pam_tau_scale_range override did NOT reach the actuator."
            )
        # Illustrative print: tau_final = tau_scale * tau_base for a few P_cmd
        # samples, using the same 1D table torque.py builds when
        # use_pressure_dependent_tau=False (2D map path skips this table, but
        # the multiplicative relationship tau_final = tau_base * scale is
        # identical regardless of which table produced tau_base).
        tau_p_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        tau_vals = [0.043, 0.045, 0.060, 0.066, 0.094, 0.131]
        print(f"[Tau Safety Check] PASSED: ch_DF/F/G.current_tau_scale == {expected} for all envs")
        for p, base in zip(tau_p_axis[::2], tau_vals[::2]):
            print(f"  illustrative: P_cmd={p} tau_base={base:.4f} "
                  f"tau_final(expected)={expected * base:.4f}  (tau_final = tau_scale * tau_base)")

    if args_cli.mask_mode != "none":
        obs = obs_mask.apply_mask(obs, args_cli.mask_mode, mask_lo_idx, mask_hi_idx)
        obs_t = obs_mask.policy_tensor(obs)
        print(f"[Mask] obs.shape={tuple(obs_t.shape)} post-mask sample obs[0, {mask_lo_idx}:{mask_lo_idx+3}]="
              f"{obs_t[0, mask_lo_idx:mask_lo_idx+3].tolist()}")

    # RNNリセット
    if hasattr(policy, "reset_memory"):
        policy.reset_memory()

    print("="*60)
    print(f" Sim-Verification Started")
    print(f" Mode: {args_cli.pattern} | BPM: {args_cli.bpm}")
    print("="*60)

    episode_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 推論
            actions = policy(obs)

            # ステップ実行 (環境がConfig通りのリズムを生成してくれる)
            obs, _, dones, _ = env.step(actions)
            if args_cli.mask_mode != "none":
                obs = obs_mask.apply_mask(obs, args_cli.mask_mode, mask_lo_idx, mask_hi_idx)
            if dones.any():
                episode_count += int(dones.sum().item())
                if episode_count >= args_cli.max_episodes:
                    print(f"[INFO] Reached max_episodes={args_cli.max_episodes} "
                          f"({episode_count} resets). Exiting...")
                    break

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()