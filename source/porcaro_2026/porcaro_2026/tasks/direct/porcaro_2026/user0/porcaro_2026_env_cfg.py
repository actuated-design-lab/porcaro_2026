# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/porcaro_2026_env_cfg.py

from __future__ import annotations
from pathlib import Path
import math
import torch
import os

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

# Porcaro imports
from .cfg.assets import ROBOT_CFG, DRUM_CFG
from .cfg.sensors import contact_forces_stick_at_drum_cfg, drum_vs_stick_cfg
from .cfg.controller_cfg import TorqueControllerCfg
from .cfg.logging_cfg import LoggingCfg, RewardLoggingCfg
from .cfg.rewards_cfg import RewardsCfg
from .cfg.actuator_cfg import PamGeometricCfg

@configclass
class Porcaro2026EnvCfg(DirectRLEnvCfg):
    """Porcaro 2026 環境用のベース設定クラス"""

    decimation: int = 4 # decimation*dt = 50Hz
    episode_length_s: float = 100.0
    
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,     # 2ms
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.0, dynamic_friction=0.0, restitution=0.8,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=3.0,
        replicate_physics=True,
    )
    
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    drum_cfg:  RigidObjectCfg  = DRUM_CFG.replace(prim_path="/World/envs/env_.*/Drum")
    
    stick_contact_cfg: ContactSensorCfg = contact_forces_stick_at_drum_cfg
    drum_contact_cfg: ContactSensorCfg = drum_vs_stick_cfg
    
    action_space: int = 3
    observation_space: int = 15
    state_space: int = 0
    dof_names: list[str] = ["Base_link_Wrist_joint", "Hand_link_Grip_joint"]

    # Model B: 有効収縮率とたわみを考慮する幾何学設定
    pam_geometric_cfg: PamGeometricCfg = PamGeometricCfg(
        natural_length=0.150,
        use_absolute_geometry = False,
        wire_slack_offsets=(0.0, 0.0, 0.0),
    )

    use_simple_rhythm: bool = False  
    simple_rhythm_mode: str = "single_8" 
    simple_rhythm_bpm: float = 120.0    
    target_hit_force: float = 20.0
    lookahead_horizon: float = 0.1
    bpm_range: tuple[float, float] = (60.0, 160.0)

    pam_tau_scale_range: tuple[float, float] = (1.0, 1.0)

    controller: TorqueControllerCfg = TorqueControllerCfg()
    logging: LoggingCfg = LoggingCfg(enabled=False)
    rewards: RewardsCfg = RewardsCfg() 
    reward_logging: RewardLoggingCfg = RewardLoggingCfg()
    events: PorcaroEventCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.rewards, "target_force_fd"):
            self.rewards.target_force_fd = self.target_hit_force


@configclass
class PorcaroEventCfg:
    randomize_mass: EventTerm = None
    randomize_material: EventTerm = None
    reset_robot_joints: EventTerm = None

def apply_domain_randomization(cfg: Porcaro2026EnvCfg):
    cfg.events = PorcaroEventCfg()
    cfg.events.randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
    )
    cfg.events.randomize_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "static_friction_range": (0.95, 1.05), "dynamic_friction_range": (0.95, 1.05), "restitution_range": (0.0, 0.0), "num_buckets": 64},
    )
    cfg.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.95, 1.05), "velocity_range": (0, 0)},
    )


# =========================================================
#  Model B (IROSで大成功したモデル) の設定クラス
# =========================================================

# --- Model B (DRなし: ベースライン用) ---
@configclass
class Porcaro2026EnvCfg_ModelB(Porcaro2026EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.controller.tau = 0.09 
        self.controller.use_pressure_dependent_tau = False # 2D Mapを使用
        self.controller.pam_viscosity = 0.0 

# --- Model B (DRあり: 実機デプロイ用最強モデル) ---
@configclass
class Porcaro2026EnvCfg_ModelB_DR(Porcaro2026EnvCfg_ModelB):
    def __post_init__(self):
        super().__post_init__()
        apply_domain_randomization(self)
        self.pam_tau_scale_range = (0.8, 1.2)