"""ral_data.py — 図スクリプトが使う集計済みCSVの読み込みと、共通の統計処理。

読むのは `paper_data/` 以下のCSVだけ。GPU も isaaclab も torch も使わない。

  paper_data/sim/sim_summary_1N.csv       sim 本評価 75ラン（A-E x 5seed x 3曲）
  paper_data/sim/sim_strikes_1N.csv       同・打点ごと
  paper_data/tau/tau_summary_1N.csv       tau スイープ 81ラン
  paper_data/hardware/hw_summary_s4_1N.csv  実機 第4セッション 105ラン
  paper_data/hardware/hw_strikes_s4_1N.csv  同・打点ごと
  paper_data/ablation/mask_{zero,noise,shuffle}_summary_1N.csv  マスク各15ラン

判定は sim・実機とも 1N（min_strike_frac=0.05 x target_ref=20N）、許容 +/-30ms で統一。
解析単位は「学習シード」（n=5）であってランではない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "paper_data"

# 条件名。sim と実機で表記が違うので、ここで一元化する。
GMD03_SIM = "gmd_03_high_bpm138"
GMD03_HW = "gmd_03_high_bpm138.mid"
DOUBLE_SIM = "double_160bpm"
DOUBLE_HW = "test_double_bpm160.mid"
SINGLE_HW = "test_single8_bpm120.mid"
GMD04_SIM = "gmd_04_extreme_bpm170"

# 本文の比較に使う条件（single8 は12打点しかなく検出力が無いので外す）
MAIN_SIM = [DOUBLE_SIM, GMD03_SIM]
MAIN_HW = [DOUBLE_HW, GMD03_HW]

MODELS = list("ABCDE")
LOOKAHEAD_S = {"A": 0.1, "B": 0.5, "C": 1.0}


# ----------------------------------------------------------------- 読み込み
def sim_summary() -> pd.DataFrame:
    return pd.read_csv(DATA / "sim" / "sim_summary_1N.csv")


def sim_strikes() -> pd.DataFrame:
    return pd.read_csv(DATA / "sim" / "sim_strikes_1N.csv")


def _hw_keys(d: pd.DataFrame) -> pd.DataFrame:
    """model_key ('RAL/B_seed3') から model と seed を復元する。"""
    d = d.copy()
    d["model"] = d["model_key"].astype(str).str.extract(r"/([A-E])_seed")[0]
    d["seed"] = d["model_key"].astype(str).str.extract(r"seed(\d)")[0].astype(int)
    return d


def hw_summary() -> pd.DataFrame:
    return _hw_keys(pd.read_csv(DATA / "hardware" / "hw_summary_s4_1N.csv"))


def hw_strikes() -> pd.DataFrame:
    return _hw_keys(pd.read_csv(DATA / "hardware" / "hw_strikes_s4_1N.csv"))


def mask_summary(mode: str) -> pd.DataFrame:
    """mode in {zero, noise, shuffle}。マスク無しの基準は sim_summary の Model C。"""
    return pd.read_csv(DATA / "ablation" / f"mask_{mode}_summary_1N.csv")


def tau_summary() -> pd.DataFrame:
    return pd.read_csv(DATA / "tau" / "tau_summary_1N.csv")


# ------------------------------------------------------------------- 集約
def seed_level(d: pd.DataFrame, task_col: str, tasks) -> pd.DataFrame:
    """条件を絞り、曲と trial をまたいでシード内で平均する。

    返り値は [model, seed, success_rate] の縦持ち。**必ずこの単位で検定する。**
    ランを単位にすると trial を多く取ったモデルが実質的に重み付けされてしまう。
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    sub = d[d[task_col].isin(tasks)]
    return sub.groupby(["model", "seed"], as_index=False)["success_rate"].mean()


def by_model(seed_df: pd.DataFrame) -> pd.DataFrame:
    """seed_level の出力を model x seed の横持ちにする。"""
    return seed_df.pivot(index="seed", columns="model", values="success_rate")


def mean_sd(seed_df: pd.DataFrame, model: str) -> tuple[float, float]:
    v = seed_df[seed_df.model == model]["success_rate"].values
    return float(v.mean()), float(v.std(ddof=1))


def ttest(seed_df: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """(差 b-a, p値)。独立2標本t検定（シード単位、n=5）。"""
    va = seed_df[seed_df.model == a]["success_rate"].values
    vb = seed_df[seed_df.model == b]["success_rate"].values
    return float(vb.mean() - va.mean()), float(stats.ttest_ind(va, vb).pvalue)


def ptext(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def jitter(n: int, w: float = 0.055) -> np.ndarray:
    """決定論的な等間隔ジッタ。乱数を使わないので図が毎回同一になる。"""
    return np.linspace(-w, w, n) if n > 1 else np.zeros(1)


def struck_only(strikes: pd.DataFrame, thr_col: str | None = "force_threshold",
                thr: float = 1.0) -> pd.DataFrame:
    """打撃が成立した打点だけに絞る。力不足で失格した打点の「誤差」は意味を持たない。"""
    if thr_col and thr_col in strikes.columns:
        return strikes[strikes.peak_force >= strikes[thr_col]]
    return strikes[strikes.peak_force >= thr]
