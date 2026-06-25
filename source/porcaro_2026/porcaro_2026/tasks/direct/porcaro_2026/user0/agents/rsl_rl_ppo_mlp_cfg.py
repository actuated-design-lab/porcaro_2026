# source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/agents/rsl_rl_ppo_mlp_cfg.py

from isaaclab.utils import configclass

# 💡 修正点1: RecurrentCfg を外し、通常の ActorCriticCfg をインポート
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoAlgorithmCfg, 
    RslRlPpoActorCriticCfg # MLP用のクラス
) 

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ★ LSTMの設定と完全に一致させる
    num_steps_per_env = 120
    max_iterations = 1500
    save_interval = 50
    
    # ★変更: 実験名が混ざらないように MLP & DRなし であることを明記
    experiment_name = "porcaro_rslrl_mlp_modelB_DR_lookahead5" 
    
    # 💡 修正点2: Policyクラスを通常のMLPバージョンに変更
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True, 
        critic_obs_normalization=True, 
        
        # LSTMの層と条件を合わせるため、同じ次元数を採用
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        
        # 💡 修正点3: RNN関連の引数 (rnn_type, rnn_hidden_dim, rnn_num_layers) は削除
    )
    
    # ★ LSTMの設定と完全に一致させる
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, 
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )