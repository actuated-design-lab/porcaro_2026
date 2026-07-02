# user1/actions/discrete_torque.py

from __future__ import annotations
import torch
from ...common.actions.torque import TorqueActionController


class DiscreteTorqueActionController(TorqueActionController):
    """
    電磁弁制御用：エージェント出力を離散値（0 or Pmax MPa）に変換するコントローラ。
    TorqueActionController を継承し、_compute_command_pressure のみ上書きする。
    """

    def __init__(self, *args, discrete_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_threshold = discrete_threshold  # この値以上なら Pmax、未満なら 0

    def _compute_command_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        # まず通常の連続値変換を行う（-1〜1 → 0〜Pmax）
        P_continuous = super()._compute_command_pressure(actions)

        # 閾値で 0 or Pmax に2値化する
        P_discrete = torch.where(
            P_continuous >= self.discrete_threshold * self.Pmax,
            torch.full_like(P_continuous, self.Pmax),   # ON: 0.6 MPa
            torch.zeros_like(P_continuous),              # OFF: 0.0 MPa
        )
        return P_discrete