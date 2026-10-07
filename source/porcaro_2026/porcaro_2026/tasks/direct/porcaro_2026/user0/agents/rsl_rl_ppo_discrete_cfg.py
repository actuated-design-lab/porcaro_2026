# user0/agents/rsl_rl_ppo_discrete_cfg.py
"""電磁弁（2値）タスク用の PPO 設定。

ネットワーク・ハイパーパラメータは連続値版と同じで、experiment_name だけを変える。
RA-L の学習ログ（logs/user0/rsl_rl/porcaro_rslrl_{lstm_modelB_DR,mlp_modelB_DR_lookahead5}/）と
同じフォルダに入ると、analysis/harness/discover.py が RA-L の Model A〜E として拾ってしまうため。
"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg as _DefaultRunnerCfg
from .rsl_rl_ppo_lstm_cfg import PPORunnerCfg as _LstmRunnerCfg
from .rsl_rl_ppo_mlp_cfg import PPORunnerCfg as _MlpRunnerCfg


@configclass
class PPORunnerCfg(_DefaultRunnerCfg):
    experiment_name = "porcaro_rslrl_lstm_modelB_DR_discrete"


@configclass
class LstmPPORunnerCfg(_LstmRunnerCfg):
    experiment_name = "porcaro_rslrl_lstm_modelB_DR_discrete"


@configclass
class MlpPPORunnerCfg(_MlpRunnerCfg):
    experiment_name = "porcaro_rslrl_mlp_modelB_DR_discrete"
