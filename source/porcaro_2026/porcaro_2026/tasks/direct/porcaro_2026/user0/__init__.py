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
    },
)
