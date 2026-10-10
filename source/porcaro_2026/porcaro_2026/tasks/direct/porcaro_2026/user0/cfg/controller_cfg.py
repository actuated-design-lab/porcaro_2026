# controller_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

from .assets import FORCE_MAP_CSV, H0_MAP_CSV

@configclass
class TorqueControllerCfg:
    """TorqueActionController の設定"""
    # 制御モード ("ep" or "pressure")
    control_mode: str = "pressure"
    
    r: float = 0.014
    L: float = 0.150
    theta_t_DF_deg: float = 7.0
    theta_t_F_deg: float = 70.0
    theta_t_G_deg: float = 45.0
    Pmax: float = 0.6 # pneumatic.py (Table I) の最大に合わせる
    tau: float = 0.09
    dead_time: float = 0.03
    N: float = 630.0 # 簡易式 Fpam_quasi_static 用 (CSVがあれば不要)
    pam_viscosity: float = 0.0
    force_map_csv: str = FORCE_MAP_CSV
    # 力マップの倍率。スカラー（3筋共通）か (DF, F, G) の組（筋ごと）。RA-L は 0.2
    force_scale: float | tuple[float, float, float] = 0.2
    h0_map_csv: str = H0_MAP_CSV
    use_pressure_dependent_tau: bool = True

    # --- PAM のヒステリシス力と、たるみの境目（以前は TorqueActionController の既定値に固定。値は同じ） ---
    pam_hys_const: float = 0.5          # c0 [N]
    pam_hys_coef_p: float = 15.0        # cP [N/MPa]
    pam_contract_gain: float = 1.5      # 収縮側の非対称係数
    pam_extend_gain: float = 1.0        # 伸長側の非対称係数
    pam_p_dot_scale: float = 100.0      # 向き d = tanh(clip(scale·dP/dt)/0.1) の scale
    transition_width: float = 0.0       # たるみ→張りの遷移幅（0 = 階段）

    # --- ヒステリシス力の向きの決め方（2026/10/10 追加。既定は従来どおり relay） ---
    #   relay: 圧力の変化率の符号で即座に切り替える（RA-L まで）
    #   play : 圧力の遊び作用素。幅 w_i [MPa] 以上戻ったときに連続に切り替わる（速さに依らない）
    pam_hys_mode: str = "relay"
    pam_hys_play_widths: tuple[float, ...] = (0.02,)    # 片側の幅 w_i [MPa]。複数並べると Prandtl–Ishlinskii 型
    pam_hys_play_weights: tuple[float, ...] | None = None   # 各要素の重み（合計 1）。None = 等分

    # --- 電磁弁（2値）モード。既定 False = 従来どおりの連続値（RA-L と同じ） ---
    use_discrete_action: bool = False   # True で指令圧力を 0 / Pmax の2値にする（common/actions/discrete_torque.py）
    discrete_threshold: float = 0.5     # 連続値の指令圧力が Pmax の何割以上で ON にするか（0〜1）
