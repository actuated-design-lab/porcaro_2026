"""ral_figstyle.py — RA-L 本文図の共通スタイル。

**ここだけ触れば全図に効く。** 個々の図は fig4_simresults.py / fig5_quanteval.py /
figS_*.py に分かれているので、レイアウトの微調整はそちらで行う。

■ フォント
  本文が IEEEtran（Times系）なので、図も Times 系に揃える。
  Windows では実物の "Times New Roman" が拾われる。無い環境では
  STIXGeneral → TeX Gyre Termes → Liberation Serif の順にフォールバックする
  （いずれも Times メトリック互換、matplotlib に同梱またはTeX Liveに同梱）。
  数式は mathtext の STIX セットで、本文と字面が揃う。

■ 配色
  Okabe-Ito（色覚多様性対応）。dataviz スキルの validate_palette.js で
  light サーフェス・全ペアを検証済み:
    A #E69F00 / B #0072B2 / C #56B4E9 / D #D55E00 / E #009E73
      [PASS] Lightness band / Chroma floor
      [PASS] CVD separation      worst all-pairs #009E73<->#D55E00 dE 11.0 (deutan)
      [PASS] Normal-vision floor worst all-pairs #D55E00<->#E69F00 dE 15.6
      [WARN] Contrast vs surface #E69F00 2.19 / #56B4E9 2.25 (<3:1)
             -> 全図で凡例＋直接ラベル＋白縁取りを付けて色単独の識別に頼らない

■ 出力
  既定は PDF（ベクタ、フォント Type42 埋め込み）。`--format png` で PNG も出せる。
  IEEEtran 2段組を想定し、1段幅 3.5in / 2段幅 7.16in。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Okabe-Ito ---
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# モデル A-E は「実体」に色を固定する（順位ではなく実体に色を付ける）
MODEL_COLORS = {
    "A": OKABE_ITO["orange"],
    "B": OKABE_ITO["blue"],
    "C": OKABE_ITO["skyblue"],
    "D": OKABE_ITO["vermillion"],
    "E": OKABE_ITO["green"],
}
MODEL_SHORT = {
    "A": "A\n0.1 s",
    "B": "B\n0.5 s",
    "C": "C\n1.0 s",
    "D": "D\nno mem.",
    "E": "E\nstack 5",
}
MODEL_LS = {"A": "-", "B": "-", "C": (0, (4, 1.6)), "D": "-", "E": (0, (1, 1.4))}

# sim / 実機 の2系列（同一エンティティの2ドメイン）
DOMAIN_COLORS = {"sim": OKABE_ITO["blue"], "real": OKABE_ITO["vermillion"]}
DOMAIN_LABEL = {"sim": "Simulation", "real": "Hardware"}

# マスク条件。ベースラインは中立インク（カテゴリ色を消費しない参照系列）
MASK_COLORS = {
    "none": "#444444",
    "zero": OKABE_ITO["blue"],
    "noise": OKABE_ITO["vermillion"],
    "shuffle": OKABE_ITO["green"],
}

# Fig.5(b) のタイミング誤差ヒストグラム専用。
# ★同じ図の中で青=sim / 橙=実機（パネルa）、緑=success / 橙=late / 灰=no strike
#   （パネルc）という意味付けを使っているので、B/C/E をこの3系統のどれかで塗ると
#   「(b)の緑=E」と「(c)の緑=success」が読者の中で混線する。
#   そこで (b) だけは意味を持たない中立色（黒・紫・濃灰）を使い、さらに線種でも
#   分けて、白黒印刷でも区別できるようにする。
TIMING_COLORS = {"B": OKABE_ITO["black"], "C": OKABE_ITO["purple"], "E": "#6b6b6b"}
TIMING_LS = {"B": "-", "C": "--", "E": "-."}

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dddddd"
BAND = "#e8e8e8"  # +/-30ms 帯

COL_W = 3.5   # IEEE 1段
DBL_W = 7.16  # IEEE 2段

# Times New Roman を先頭に置く。Windows では実物が、それ以外では互換フォントが当たる。
SERIF_STACK = [
    "Times New Roman",
    "STIXGeneral",
    "TeX Gyre Termes",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]


def apply_style(base: float = 8.0) -> None:
    """rcParams を設定する。`base` は本文相当のポイント数。"""
    plt.rcParams.update({
        "pdf.fonttype": 42,      # TrueType 埋め込み（IEEE の PDF 要件）
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix",   # 数式も Times 系で揃える
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
    })


def tidy(ax, ygrid: bool = True) -> None:
    """余計な枠と目盛りを落として、グリッドを背面の薄い線だけにする。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)


def panel_tag(ax, tag: str, x: float = -0.16, y: float = 1.14) -> None:
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=plt.rcParams["font.size"] + 1,
            fontweight="bold", color=INK, va="top", ha="left")


def base_argparser(description: str) -> argparse.ArgumentParser:
    """図スクリプト共通のコマンドライン引数。"""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--outdir", default="figures", help="出力先ディレクトリ")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "both"],
                    help="出力形式（既定 pdf。png はレビュー用の 300 dpi）")
    ap.add_argument("--fontsize", type=float, default=8.0, help="本文相当のポイント数")
    return ap


def save(fig, outdir: str | Path, stem: str, fmt: str = "pdf") -> list[Path]:
    """PDF（と必要なら PNG）で書き出してパスを返す。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    if fmt in ("pdf", "both"):
        p = outdir / f"{stem}.pdf"
        fig.savefig(p)
        out.append(p)
    if fmt in ("png", "both"):
        p = outdir / f"{stem}.png"
        fig.savefig(p, dpi=300)
        out.append(p)
    plt.close(fig)
    for p in out:
        print(f"[saved] {p}  ({p.stat().st_size/1024:.0f} KB)")
    return out
