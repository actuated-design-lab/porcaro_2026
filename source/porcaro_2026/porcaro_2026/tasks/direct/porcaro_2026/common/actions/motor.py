# source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026v1/actions/motor.py

from __future__ import annotations
import torch
from .base import ActionController, RobotLike


class MotorController(ActionController):
    """
    Elbow用モータ位置制御器

    RL action[0] をElbowの目標関節角度へ変換し、
    PD制御によってElbowのトルクを計算する。

    実際のrobot.set_joint_effort_target()はここでは行わず、
    計算したトルクをenv.pyへ返す。
    """

    def __init__(
        self,
        dt_ctrl: float,
        target_min_deg: float = -70.0,
        target_max_deg: float = 20.0,
        kp: float = 0.0,
        kd: float = 0.0,
        effort_limit: float = 0.0,
    ):
        self.dt_ctrl = float(dt_ctrl)

        # Elbowの目標角度範囲 [deg]
        self.target_min_deg = float(target_min_deg)
        self.target_max_deg = float(target_max_deg)

        # PDゲイン
        self.kp = float(kp)
        self.kd = float(kd)

        # 最大トルク
        self.effort_limit = float(effort_limit)

        # reset()でdeviceへ移すためのTensor
        self.target_min_rad = None
        self.target_max_rad = None

        self._last_telemetry: dict | None = None

    def reset(
        self,
        n_envs: int,
        device: str | torch.device,
    ):
        """全環境の状態を初期化する。"""

        device = torch.device(device)

        self.target_min_rad = torch.tensor(
            self.target_min_deg * torch.pi / 180.0,
            device=device,
            dtype=torch.float32,
        )

        self.target_max_rad = torch.tensor(
            self.target_max_deg * torch.pi / 180.0,
            device=device,
            dtype=torch.float32,
        )

        self._last_telemetry = None

    def reset_idx(self, env_ids: torch.Tensor):
        """指定された環境の状態をリセットする。"""

        if self._last_telemetry is not None:
            for key, value in self._last_telemetry.items():
                if isinstance(value, torch.Tensor):
                    value[env_ids] = 0.0

    def _action_to_target(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        RL action [-1, 1] をElbow目標角度[rad]へ変換する。
        """

        action = torch.nan_to_num(action, nan=0.0)
        action = torch.clamp(action, -1.0, 1.0)

        # [-1,1] → [0,1]
        normalized = (action + 1.0) * 0.5

        # [0,1] → [target_min_rad, target_max_rad]
        target_rad = (
            self.target_min_rad
            + normalized
            * (self.target_max_rad - self.target_min_rad)
        )

        return target_rad

    @torch.no_grad()
    def apply(
        self,
        *,
        action: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        joint_id: int,
    ) -> torch.Tensor:
        """
        Elbowの目標角度からPD制御トルクを計算して返す。

        Parameters
        ----------
        action:
            RLから与えられるElbow用アクション。
            shape = [num_envs]

        q:
            ロボットの関節角度。
            shape = [num_envs, num_joints]

        qd:
            ロボットの関節角速度。
            shape = [num_envs, num_joints]

        joint_id:
            Elbowのjoint index。

        Returns
        -------
        torch.Tensor
            Elbowに与えるトルク。
            shape = [num_envs]
        """

        n_envs = int(q.shape[0])

        # RL action → 目標角度
        target_rad = self._action_to_target(action)

        # 現在角度・角速度
        current_rad = q[:, joint_id]
        current_vel_rad = qd[:, joint_id]

        # 位置誤差
        position_error = target_rad - current_rad

        # PD制御
        torque = (
            self.kp * position_error
            - self.kd * current_vel_rad
        )

        # トルク制限
        if self.effort_limit > 0.0:
            torque = torch.clamp(
                torque,
                min=-self.effort_limit,
                max=self.effort_limit,
            )

        self._last_telemetry = {
            "target_rad": target_rad.clone(),
            "actual_rad": current_rad.clone(),
            "actual_vel_rad": current_vel_rad.clone(),
            "position_error_rad": position_error.clone(),
            "torque": torque.clone(),
        }

        return torque

    def get_last_telemetry(self):
        return self._last_telemetry