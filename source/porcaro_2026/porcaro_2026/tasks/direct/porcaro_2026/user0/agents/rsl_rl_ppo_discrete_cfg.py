# user0/agents/rsl_rl_ppo_discrete_cfg.py
"""電磁弁（2値）タスク用の PPO 設定。

ネットワークは連続値版と同じ。変えるのは次の2つだけ:
- experiment_name: RA-L の学習ログ（logs/user0/rsl_rl/porcaro_rslrl_{lstm_modelB_DR,mlp_modelB_DR_lookahead5}/）
  と同じフォルダに入ると、analysis/harness/discover.py が RA-L の Model A〜E として拾ってしまうため分ける。
- entropy_coef: 2値では方策の平均が閾値（a=0）から離れると ON/OFF がほぼ反転しなくなり、勾配が消えて固まる。
  エントロピー項を連続値版（0.002）より強くして、ノイズが縮みきらないようにする。
"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg as _DefaultRunnerCfg
from .rsl_rl_ppo_lstm_cfg import PPORunnerCfg as _LstmRunnerCfg
from .rsl_rl_ppo_mlp_cfg import PPORunnerCfg as _MlpRunnerCfg

DISCRETE_ENTROPY_COEF = 0.01


@configclass
class PPORunnerCfg(_DefaultRunnerCfg):
    experiment_name = "porcaro_rslrl_lstm_dr_discrete"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = DISCRETE_ENTROPY_COEF


@configclass
class LstmPPORunnerCfg(_LstmRunnerCfg):
    experiment_name = "porcaro_rslrl_lstm_dr_discrete"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = DISCRETE_ENTROPY_COEF


@configclass
class MlpPPORunnerCfg(_MlpRunnerCfg):
    experiment_name = "porcaro_rslrl_mlp_dr_discrete"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = DISCRETE_ENTROPY_COEF
