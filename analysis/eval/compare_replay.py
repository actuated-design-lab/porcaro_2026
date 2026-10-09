"""
compare_replay.py — 開ループ再生（scripts/replay_open_loop.py）の sim 出力と実機ログを並べて比べる

入力:
  data/user0/jfps2026/replay/sim_<動作>_<圧力モデル>.csv（200 Hz）
  ../jetson_project/test_signals/<動作>.csv（指令, 50 Hz）と data_<動作>_<数字>.csv（実機ログ, 200 Hz）
  実機ログは scripts/real_log.py で読む（DF/F 圧力列の入れ替わりを戻し、flag のエコーで指令に時刻を合わせる。
  replay_open_loop.py の measured モードと同じ処理）

指標（動作 × 圧力モデル。各信号の最初の 2 s の保持区間は sim が 0 MPa から立ち上がるので --skip_s で除く）:
  角度（wrist, grip）: RMSE、平均ずれ（sim − 実機, 静的）、平均ずれを引いた RMSE（動的）、
                       相互相関の遅れ（正 = sim が遅い。相関 r < 0.5 か探索幅 ±0.25 s の端なら空欄）
  圧力（DF, F, G）  : NRMSE [%] = RMSE / 実機の幅 × 100（jetson_project analysis/itv/pmeval.py と同じ定義）。
                       指令が一定のチャネルは幅がノイズだけになるので空欄
ワイヤー外れの洗い出し（除外はしない。一覧だけ出す）:
  実機の角度の 1 サンプルの跳び、角度が 0.2 deg 以内に張り付いている最長区間と、その区間で
  measuredV2（実測圧力を入れた sim。なければ measured）が動いた幅。sim が動くのに実機が動かない区間が怪しい

出力（--out, 既定 data/user0/jfps2026/replay/compare/）:
  metrics.csv            動作 × 圧力モデルの全指標（1 行 = 1 組）
  summary.md             主な指標の表（動作 × 圧力モデル）
  wire_screen.csv        ワイヤー外れの洗い出し
  angles_<動作>.png      角度波形（実機と measured / measuredV2 / orificeV2 / shaped / pilot_leak）
  overview_measured.png  全動作の実機と measuredV2（なければ measured）。ワイヤー外れの目視確認用

使い方:
  python analysis/eval/compare_replay.py
  python analysis/eval/compare_replay.py --jetson ~/my_isaaclab_projects/jetson_project --skip_s 2.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import real_log as RL  # noqa: E402

MOTIONS = ["tm_A_quasistatic", "tm_B_steps", "tm_C_sine", "tm_D_antagonist",
           "tm_E_dbl160_seed2", "tm_E_dbl160_seed3", "tm_E_gmd138_seed2", "tm_E_gmd138_seed3"]
MODELS = ["table", "lag", "orifice", "orificeV2", "shaped", "pilot_leak", "measured", "measuredV2"]
JOINTS = ["wrist", "grip"]
CH = ["DF", "F", "G"]
DT = 0.005
MAX_LAG_S = 0.25
MIN_XCORR_R = 0.5

# 波形図に載せるモデルと色（固定順。実機は文字色の黒）
PLOT_MODELS = [("measured", "#2a78d6"), ("measuredV2", "#eb6834"), ("orificeV2", "#1baf7a"), ("shaped", "#eda100"),
               ("pilot_leak", "#e87ba4")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def real_log_path(jetson, motion, replay_dir):
    """measured の json に書かれた実機ログを優先し、なければ data_<動作>_<数字>.csv を探す（_drum は除く）"""
    meta = os.path.join(replay_dir, f"sim_{motion}_measured.json")
    if os.path.exists(meta):
        p = json.load(open(meta)).get("real_log")
        if p:
            p = os.path.join(jetson, "test_signals", os.path.basename(p))
            if os.path.exists(p):
                return p
    c = [f for f in glob.glob(os.path.join(jetson, "test_signals", f"data_{motion}_*.csv"))
         if os.path.basename(f)[len(f"data_{motion}_"):-4].isdigit()]
    if len(c) != 1:
        raise SystemExit(f"{motion}: 実機ログが {len(c)} 本見つかった: {c}")
    return c[0]


def load_real(jetson, motion, replay_dir):
    cmd = pd.read_csv(os.path.join(jetson, "test_signals", f"{motion}.csv"))[
        ["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values
    path = real_log_path(jetson, motion, replay_dir)
    A, lag = RL.align_real_log(pd.read_csv(path), cmd, DT,
                               columns=RL.PRES_COLS + ("wrist_angle_deg", "grip_angle_deg"))
    real = pd.DataFrame(A, columns=[f"P_{c}" for c in CH] + ["wrist", "grip"])
    real.insert(0, "time", np.arange(len(real)) * DT)
    c200 = cmd[np.clip((real["time"].values / RL.CMD_DT + 1e-9).astype(int), 0, len(cmd) - 1)]
    return real, c200, path, lag


def xcorr_lag_ms(sim, real):
    """sim が実機より何 ms 遅れているか（平均を引いた相互相関の最大, ±MAX_LAG_S）と、そのときの相関係数

    相関が弱い（r < MIN_XCORR_R）か、最大が探索幅の端に来たときは遅れを NaN にする（grip のように sim がほぼ動かないとき）
    """
    a = sim - sim.mean(); b = real - real.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan, np.nan
    K = int(MAX_LAG_S / DT); n = len(a)
    cc = {k: np.dot(a[max(k, 0): n + min(k, 0)], b[max(-k, 0): n - max(k, 0)]) for k in range(-K, K + 1)}
    best = max(cc, key=cc.get)
    r = cc[best] / (np.linalg.norm(a) * np.linalg.norm(b))
    if r < MIN_XCORR_R or abs(best) == K:
        return np.nan, r
    return best * DT * 1000, r


def metrics(sim, real, cmd, skip):
    n = min(len(sim), len(real)); i0 = int(round(skip / DT))
    s, r, c = sim.iloc[i0:n], real.iloc[i0:n], cmd[i0:n]
    row = {}
    for j in JOINTS:
        e = s[f"{j}_angle_deg"].values - r[j].values
        row[f"{j}_rmse"] = np.sqrt(np.mean(e ** 2))
        row[f"{j}_bias"] = e.mean()
        row[f"{j}_rmse_dyn"] = e.std()
        row[f"{j}_lag_ms"], row[f"{j}_xcorr_r"] = xcorr_lag_ms(s[f"{j}_angle_deg"].values, r[j].values)
    for k, ch in enumerate(CH):
        m = r[f"P_{ch}"].values
        row[f"P_{ch}_nrmse"] = (np.sqrt(np.mean((s[f"P_out_{ch}"].values - m) ** 2)) / np.ptp(m) * 100
                                if np.ptp(c[:, k]) > 0.01 else np.nan)
    return row


def wire_screen(motion, real, cmd, meas, skip):
    """実機の角度の跳びと張り付き。張り付いた区間で measured の sim が動いた幅を並べる"""
    rows = []
    i0 = int(round(skip / DT))
    for j in JOINTS:
        x = real[j].values
        dx = np.abs(np.diff(x[i0:]))
        # 張り付き区間（0.2 deg 以内に留まる 0.3 s 以上の区間）を全部拾う
        flats = []; a = i0
        for i in range(i0 + 1, len(x) + 1):
            if i == len(x) or abs(x[i] - x[a]) > 0.2:
                if (i - a) * DT >= 0.3:
                    flats.append((a, i - 1))
                a = i
        longest = max(flats, key=lambda f: f[1] - f[0], default=(i0, i0))
        row = dict(motion=motion, joint=j, real_min=x[i0:].min(), real_max=x[i0:].max(),
                   max_jump_deg=dx.max(), max_jump_t=(i0 + np.argmax(dx) + 1) * DT,
                   longest_flat_s=(longest[1] - longest[0]) * DT, longest_flat_from=longest[0] * DT)
        if meas is not None:
            y = meas[f"{j}_angle_deg"].values[:len(x)]
            # 張り付き区間のうち、measured の sim がいちばん動いた区間（実機だけ止まっている = 怪しい）
            a, b = max(flats, key=lambda f: np.ptp(y[f[0]:f[1] + 1]), default=(i0, i0))
            row.update(suspect_flat_from=a * DT, suspect_flat_to=b * DT, suspect_real_deg=x[a:b + 1].mean(),
                       suspect_measured_range=np.ptp(y[a:b + 1]),
                       suspect_cmd_range=" / ".join(f"{v:.2f}" for v in np.ptp(cmd[a:b + 1], axis=0)))
            e = y[i0:] - x[i0:]
            # 30 s ずつ区切った区間ごとの measured − 実機の RMSE（途中から急に悪くなるランを見つける）
            w = int(30 / DT)
            row["measured_rmse_by_30s"] = " / ".join(f"{np.sqrt(np.mean(e[k:k + w] ** 2)):.1f}"
                                                     for k in range(0, len(e), w) if len(e[k:k + w]) > w // 3)
        rows.append(row)
    return rows


def fmt_table(df, col, motions, models, fmt):
    piv = df.pivot(index="motion", columns="model", values=col).reindex(index=motions, columns=models)
    head = "| 動作 | " + " | ".join(models) + " |\n|---|" + "---:|" * len(models) + "\n"
    body = "".join("| " + m + " | " + " | ".join("" if pd.isna(v) else fmt.format(v) for v in piv.loc[m]) + " |\n"
                   for m in motions)
    return head + body


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_angles(motion, real, sims, out, skip, window=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = real["time"].values
    sel = np.ones_like(t, bool) if window is None else (t >= window[0]) & (t <= window[1])
    for ax, j, name in zip(axes, JOINTS, ("wrist [deg]", "grip [deg]")):
        ax.plot(t[sel], real[j].values[sel], color=INK, lw=2.0, label="実機")
        for mdl, col in PLOT_MODELS:
            if mdl in sims:
                s = sims[mdl]; n = min(len(s), len(t)); ts = s["time"].values[:n]; ss = sel[:n]
                ax.plot(ts[ss], s[f"{j}_angle_deg"].values[:n][ss], color=col, lw=1.4, label=mdl)
        ax.axvspan(t[sel][0], max(skip, t[sel][0]), color=GRID, alpha=0.6, lw=0)
        ax.set_ylabel(name, color=INK2, fontsize=9); _style(ax)
    axes[0].legend(ncol=6, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18))
    axes[1].set_xlabel("time [s]（灰色 = 指標から除いた保持区間）", color=INK2, fontsize=9)
    fig.suptitle(f"{motion}：関節角度（実機と各圧力モデル, 開ループ再生）", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_overview(reals, meas, out):
    fig, axes = plt.subplots(len(MOTIONS), 2, figsize=(14, 2.0 * len(MOTIONS)))
    for i, m in enumerate(MOTIONS):
        for k, j in enumerate(JOINTS):
            ax = axes[i, k]; r = reals[m]
            ax.plot(r["time"], r[j], color=INK, lw=1.2, label="実機")
            if meas.get(m) is not None:
                s = meas[m]; ax.plot(s["time"], s[f"{j}_angle_deg"], color=PLOT_MODELS[1][1], lw=1.0, label="measuredV2 / measured")
            ax.set_title(f"{m}  {j}", fontsize=8, color=INK, loc="left"); _style(ax)
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--replay_dir", default=os.path.join(ROOT, "data/user0/jfps2026/replay"))
    p.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    p.add_argument("--out", default=None, help="既定: <replay_dir>/compare")
    p.add_argument("--skip_s", type=float, default=2.0, help="指標から除く最初の区間 [s]（信号の保持区間）")
    args = p.parse_args()
    out = args.out or os.path.join(args.replay_dir, "compare")
    os.makedirs(out, exist_ok=True)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]

    rows, wires, reals, meas, missing = [], [], {}, {}, []
    for m in MOTIONS:
        real, cmd, path, lag = load_real(args.jetson, m, args.replay_dir)
        reals[m] = real
        sims = {}
        for mdl in MODELS:
            f = os.path.join(args.replay_dir, f"sim_{m}_{mdl}.csv")
            if not os.path.exists(f):
                missing.append(f"sim_{m}_{mdl}.csv"); continue
            sims[mdl] = pd.read_csv(f)
            rows.append(dict(motion=m, model=mdl, **metrics(sims[mdl], real, cmd, args.skip_s)))
        # ワイヤー外れの洗い出しと全体図は、実測圧力にフィルタをかけた measuredV2 を優先（なければ measured）
        meas_name = "measuredV2" if "measuredV2" in sims else "measured"
        meas[m] = sims.get(meas_name)
        wires += [dict(w, reference=meas_name) for w in wire_screen(m, real, cmd, meas[m], args.skip_s)]
        print(f"{m}: 実機ログ {os.path.basename(path)}（エコーのずれ {lag * 5} ms）, sim {len(sims)} 本")
        if m in ("tm_C_sine", "tm_E_dbl160_seed2"):
            plot_angles(m, real, sims, os.path.join(out, f"angles_{m}.png"), args.skip_s)
        if m == "tm_C_sine":
            plot_angles(m, real, sims, os.path.join(out, f"angles_{m}_zoom.png"), args.skip_s, window=(40, 60))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics.csv"), index=False, float_format="%.4g")
    pd.DataFrame(wires).to_csv(os.path.join(out, "wire_screen.csv"), index=False, float_format="%.4g")
    plot_overview(reals, meas, os.path.join(out, "overview_measured.png"))

    md = [f"# 開ループ再生：sim と実機の比較\n\n`python analysis/eval/compare_replay.py`（skip_s = {args.skip_s} s）の出力。"
          "sim − 実機。遅れは正 = sim が遅い（相関 r < 0.5 か探索幅 ±0.25 s の端なら空欄）。圧力 NRMSE は指令が一定のチャネルを空欄にしている。\n"]
    for j in JOINTS:
        for col, title, fmt in ((f"{j}_rmse", "RMSE [deg]", "{:.1f}"), (f"{j}_bias", "平均ずれ（静的） [deg]", "{:+.1f}"),
                                (f"{j}_rmse_dyn", "平均ずれを引いた RMSE（動的） [deg]", "{:.1f}"),
                                (f"{j}_lag_ms", "相互相関の遅れ [ms]", "{:+.0f}")):
            md.append(f"\n## {j}：{title}\n\n" + fmt_table(df, col, MOTIONS, MODELS, fmt))
    for ch in CH:
        md.append(f"\n## 圧力 {ch}：NRMSE [%]\n\n" + fmt_table(df, f"P_{ch}_nrmse", MOTIONS, MODELS, "{:.1f}"))
    if missing:
        md.append("\n## 見つからなかった sim 出力\n\n" + "\n".join(f"- {x}" for x in missing) + "\n")
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("".join(md))
    print(f"保存: {out}/ (metrics.csv, summary.md, wire_screen.csv, angles_*.png, overview_measured.png)")
    if missing:
        print(f"見つからなかった sim 出力: {len(missing)} 本")


if __name__ == "__main__":
    main()
