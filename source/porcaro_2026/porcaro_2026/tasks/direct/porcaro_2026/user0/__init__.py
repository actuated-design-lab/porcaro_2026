# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents

# ======================================================================
# Model B: ヒステリシス・たわみ考慮モデル (IROS 2026) - user0
# ======================================================================

# --- Model B (DRなし) ---
gym.register(
    id="Template-Porcaro-2026-ModelB-user0",
    entry_point=f"{__name__}.porcaro_2026_env:Porcaro2026Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.porcaro_2026_env_cfg:Porcaro2026EnvCfg_ModelB",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lstm_cfg:PPORunnerCfg",
        "rsl_rl_mlp_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_mlp_cfg:PPORunnerCfg",
    },
)

# --- Model B (DRあり: 推奨) ---
gym.register(
    id="Template-Porcaro-2026-ModelB-DR-user0",
    entry_point=f"{__name__}.porcaro_2026_env:Porcaro2026Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.porcaro_2026_env_cfg:Porcaro2026EnvCfg_ModelB_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lstm_cfg:PPORunnerCfg",
        "rsl_rl_mlp_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_mlp_cfg:PPORunnerCfg",
    },
)

# 電磁弁（2値）版。Model B DR と同じ設定で、指令圧力だけ 0 / Pmax にする。
# 学習ログは porcaro_rslrl_*_modelB_DR_discrete/ に分かれる（RA-L のログと混ざらない）。
gym.register(
    id="Template-Porcaro-2026-ModelB-DR-Discrete-user0",
    entry_point=f"{__name__}.porcaro_2026_env:Porcaro2026Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.porcaro_2026_env_cfg:Porcaro2026EnvCfg_ModelB_DR_Discrete",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_discrete_cfg:PPORunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_discrete_cfg:LstmPPORunnerCfg",
        "rsl_rl_mlp_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_discrete_cfg:MlpPPORunnerCfg",
    },
)
