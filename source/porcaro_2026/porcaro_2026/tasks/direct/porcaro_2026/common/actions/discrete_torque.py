# common/actions/discrete_torque.py
"""電磁弁（ON/OFF）を想定した2値圧力指令のコントローラ。

user1（王さん）の user1/actions/discrete_torque.py を共通化したもの。中身は同じ。
TorqueActionController を継承し、指令圧力の計算だけを上書きする:
連続値で計算した指令圧力が Pmax × discrete_threshold 以上なら Pmax、未満なら 0。
2値化は指令圧力に対して行うので、その後の PAM の遅れ・時定数（PAMChannel）はそのままかかる。

使い方: 各 user の controller cfg で use_discrete_action=True にすると、環境がこのクラスを使う。
"""
from __future__ import annotations

import torch

from .torque import TorqueActionController


class DiscreteTorqueActionController(TorqueActionController):
    """エージェント出力を 0 / Pmax の2値の指令圧力に変換するコントローラ。"""

    def __init__(self, *args, discrete_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= discrete_threshold <= 1.0:
            raise ValueError(f"discrete_threshold は 0〜1（Pmax に対する割合）: {discrete_threshold}")
        self.discrete_threshold = float(discrete_threshold)

    def _compute_command_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        # まず通常の連続値変換（-1〜1 → 0〜Pmax）
        P_continuous = super()._compute_command_pressure(actions)
        # 閾値で 0 / Pmax に2値化する
        return torch.where(
            P_continuous >= self.discrete_threshold * self.Pmax,
            torch.full_like(P_continuous, self.Pmax),
            torch.zeros_like(P_continuous),
        )
