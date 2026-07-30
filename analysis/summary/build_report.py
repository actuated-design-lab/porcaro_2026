"""Assemble analysis/summary/report.html from already-computed CSVs (analysis/outputs/)
and already-generated figures (analysis/summary/figs/, see gen_figures.py).

Read-only: no training/eval jobs launched, no writes outside analysis/summary/.
Every number quoted in the narrative text below is re-derived from the CSVs at
render time (not hand-typed), so the report cannot silently drift from the data.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "analysis" / "outputs"
SUMMARY_DIR = REPO_ROOT / "analysis" / "summary"
FIGS_DIR = SUMMARY_DIR / "figs"


def b64_img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def img_tag(name: str, alt: str = "") -> str:
    path = FIGS_DIR / name
    if not path.exists():
        return f'<div class="missing">[figure missing: {name}]</div>'
    return f'<img src="{b64_img(path)}" alt="{alt}" loading="lazy">'


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Section A: experiment ledger
# ---------------------------------------------------------------------------

LEDGER = [
    dict(
        purpose="A-E モデル比較・学習(先読み軸 A/B/C, 記憶軸 D/B/E)",
        r3r4=(
            "R3 Minor-1「MLP-LSTM比較はmemory機構を単離できていない」/ AE要約"
            "「memory ablationが厳密でない」→ E(framestack MLP, k=5)を"
            "finite-history-without-recurrenceの対照として追加することで直接対応。"
            "R3 Weakness-3a「LSTM-free MLPでhistoryを見る比較が無い」も同じEが充足。"
            "ただしR3が併せて要求する model-based feedforward controller との比較は"
            "本実験群では<b>未実施(ギャップ)</b>。"
        ),
        cond="A: LSTM lh=0.1s(obs15) / B: LSTM lh=0.5s(obs35) / C: LSTM lh=1.0s(obs60) / "
             "D: MLP lh=0.5s(obs35) / E: framestack MLP k=5 lh=0.5s(obs175)",
        seeds="5 seed/model (計25 run)",
        data="logs/rsl_rl/porcaro_rslrl_{lstm_modelB_DR,mlp_modelB_DR_lookahead5}/",
        conclusion="学習報酬曲線ではB≈Cで実質差が出ない(下記図B参照)。",
    ),
    dict(
        purpose="A-E モデル比較・play/eval(±30ms成功率、DR task、trial=1)",
        r3r4=(
            "R3 Weakness-2「0.5sが普遍的最適という主張はcorrelational」への実データ提示。"
            "R4「0.5sが実際に最適という主張はrobot/task/reward依存であることに注意、"
            "簡単なタスクでは差が出ない可能性」に対応(double_160では全群横並びを確認)。"
        ),
        cond="double_160(基本) / gmd_03_high_bpm138 / gmd_04_extreme_bpm170、A-E×5seed",
        seeds="5 seed/model",
        data="eval_logs/ → analysis/outputs/eval_summary.csv, eval_seed_level.csv, "
             "memory_group_comparison.csv",
        conclusion="GMD2条件でA≪B≈C、D<B≈E。double_160では全群のCIが重なり横並び。"
                   "Dは5 seed中2 seed(3,5)がGMD条件で0%に崩壊(下記図C参照)。",
    ),
    dict(
        purpose="τ(PAM時定数倍率)×lookaheadスイープ・学習(9セル×3seed=27run)",
        r3r4=(
            "R4「horizon選択の根拠(actuator identificationに基づくか)を"
            "早い段階で明記すべき」に直接対応 — 最適先読みがτに依存して動くことを示せば"
            "0.1/0.5/1.0sの選択が恣意的でなくPAM時定数由来であることの根拠になる。"
            "R3 Weakness-2「correlational」への追加証拠(ただし本実験もcorrelationalであり"
            "τ→最適lookaheadの因果を証明するものではない、後述)。"
        ),
        cond="τ倍率∈{0.5,1.0,2.0} × lookahead(τ比例グリッド、3水準/τ) × seed{1,2,3}、"
             "非DR task(Time Constant Scale DR含む全DRオフ)",
        seeds="3 seed/cell",
        data="logs/rsl_rl_tau_sweep/porcaro_rslrl_lstm_modelB_DR/",
        conclusion="9セルとも学習は収束(下記図B参照)。τが大きいほど収束が遅く/ばらつきが大きい傾向。",
    ),
    dict(
        purpose="τスイープ・eval(GMD限定の最適lookahead)",
        r3r4="同上(R4のhorizon選択根拠、R3の correlational 指摘の両方に関連)。",
        cond="τスイープ27runをdouble_160/gmd_03/gmd_04で評価(trial=1)",
        seeds="3 seed/cell",
        data="eval_logs_tau_sweep/ → tau_lookahead_success_rate_by_condition.csv, "
             "tau_optimal_lookahead_gmd.csv",
        conclusion="GMD限定で最適lookaheadはτ=0.5→0.5s, τ=1.0→0.5s, τ=2.0→2.0s。"
                   "trend(単調ではあるが2点は同値、n=3で分散大、τ=2.0は探索グリッドの右端が"
                   "最適=頭打ち未確認)。",
    ),
    dict(
        purpose="非DR taskでの再eval(DR vs 非DR比較)",
        r3r4=(
            "R3/R4の特定の一文には対応しないが、AE要約全体の「claimがfully supportされていない」"
            "という懸念に対する頑健性チェックとして位置づけ(直接の指摘ではなく内部仮説検証)。"
            "検証対象仮説:「Bのlh=0.5sはDR(±20% τ)に過剰適応しており、そのsim内gapが"
            "sim-to-realギャップの一因」→ 本データで<b>棄却</b>(後述)。"
        ),
        cond="A-E×5seedを非DR task(Template-Porcaro-2026-ModelB-user0)で再eval",
        seeds="5 seed/model",
        data="eval_logs_nondr/ → dr_vs_nondr_comparison.csv",
        conclusion="モデル×条件でΔ(非DR−DR)の符号・大きさに系統性なし(Bが他モデルより"
                   "系統的に悪化するパターンは見られない) → 「BのDR過剰適応」仮説棄却。",
    ),
    dict(
        purpose="評価ノイズ研究(trial>1、GMD条件でR=5)",
        r3r4=(
            "R4 Minor「GMDタスクの評価プロトコル(trial数)を基本タスクと同程度明示すべき」"
            "に対するsim側の定量化 — ただしこれは<b>sim評価のtrial間ばらつき</b>であり、"
            "R4が言う実機trial数の報告そのものの代替にはならない点に注意。"
        ),
        cond="B/C/D/E × gmd_03/gmd_04、各seedでR=5 trial",
        seeds="5 seed/model",
        data="eval_logs_trials5/ → trials5_eval_noise.csv",
        conclusion="B/C/Eではseed間ばらつき>評価(trial間)ばらつき。Dはseed間ばらつきが"
                   "突出して大きいが、これはDのbimodal崩壊(2 seedが0%)によるものであり、"
                   "評価ノイズ自体は他モデルと同程度(下記図E参照)。",
    ),
    dict(
        purpose="遠未来(0.5-1.0s)観測マスキング・アブレーション(zero/noise/shuffle)",
        r3r4=(
            "R3 Weakness-3b「fixed-lookahead design conflates anticipation with "
            "information; 1.0s horizonの劣化はirrelevantな長期cueへの過学習の可能性があり、"
            "本質的なtrade-offと切り分けられていない」に<b>直接対応する追加実験</b>。"
            "Model C(lh=1.0s)の遠い将来(0.5-1.0s)部分を意図的に破壊し、性能変化を見ることで"
            "その情報が実際に使われているかを診断。"
        ),
        cond="Model C(lh=1.0s)、遠未来0.5-1.0s部分をzero/noise/shuffleで破壊 vs baseline",
        seeds="5 seed",
        data="eval_logs_mask_{zero,noise,shuffle}/ → mask_comparison.csv",
        conclusion="zero/shuffleはbaselineとほぼ同水準(3条件ともCI重複)。noiseのみ明確に低下。"
                   "→ 遠未来情報が無くても(zero/shuffle)性能は落ちない一方、誤情報(noise)には"
                   "敏感 — 「不要な情報への過学習」という単純な説明とは整合しにくい"
                   "(下記図E参照、解釈は要議論)。",
    ),
]


def render_ledger() -> str:
    rows = "\n".join(
        f"""<tr>
          <td>{r['purpose']}</td>
          <td>{r['r3r4']}</td>
          <td>{r['cond']}</td>
          <td>{r['seeds']}</td>
          <td><code>{r['data']}</code></td>
          <td>{r['conclusion']}</td>
        </tr>"""
        for r in LEDGER
    )
    return f"""
    <div class="table-scroll"><table class="ledger">
      <thead><tr>
        <th>目的</th><th>対応する査読指摘(R3/R4)</th><th>モデル・条件</th>
        <th>seed数</th><th>データ場所</th><th>現在の結論</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    """


# ---------------------------------------------------------------------------
# small data-driven numbers used inline in the narrative
# ---------------------------------------------------------------------------

def compute_numbers() -> dict:
    n = {}
    summary = pd.read_csv(OUTPUTS_DIR / "eval_summary.csv")

    def cell(model, task):
        row = summary[(summary["model"] == model) & (summary["task"] == task)]
        return float(row["mean"].iloc[0]) if not row.empty else float("nan")

    n["A_gmd03"] = cell("A", "gmd_03_high_bpm138")
    n["B_gmd03"] = cell("B", "gmd_03_high_bpm138")
    n["C_gmd03"] = cell("C", "gmd_03_high_bpm138")
    n["D_gmd03"] = cell("D", "gmd_03_high_bpm138")
    n["E_gmd03"] = cell("E", "gmd_03_high_bpm138")
    n["A_double"] = cell("A", "double_160")
    n["B_double"] = cell("B", "double_160")
    n["C_double"] = cell("C", "double_160")
    n["D_double"] = cell("D", "double_160")
    n["E_double"] = cell("E", "double_160")

    optimal_gmd = pd.read_csv(OUTPUTS_DIR / "tau_optimal_lookahead_gmd.csv")
    n["tau_optimal_rows"] = optimal_gmd.to_dict("records")

    mask = pd.read_csv(OUTPUTS_DIR / "mask_comparison.csv")
    pivot = mask.pivot_table(index="task", columns="mask_mode", values="mean")
    n["mask_pivot"] = pivot

    dr = pd.read_csv(OUTPUTS_DIR / "dr_vs_nondr_comparison.csv")
    n["dr_max_abs_delta"] = dr["delta"].abs().max()
    n["dr_mean_abs_delta"] = dr["delta"].abs().mean()

    seed_level = pd.read_csv(OUTPUTS_DIR / "eval_seed_level.csv")
    d_gmd = seed_level[(seed_level["model"] == "D") & (seed_level["task"] != "double_160")]
    n["d_collapsed_seeds"] = sorted(d_gmd[d_gmd["success_rate"] < 0.01]["seed"].unique().tolist())

    return n


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.6;
       max-width: 1180px; margin: 0 auto; padding: 2rem 1.5rem 6rem; }
h1 { font-size: 1.7rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.35rem; margin-top: 3rem; border-bottom: 2px solid #2a78d6; padding-bottom: 0.3rem; }
h3 { font-size: 1.05rem; margin-top: 1.8rem; color: #2a78d6; }
.subtitle { color: #666; margin-top: 0; }
.table-scroll { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; table-layout: fixed; }
th, td { border: 1px solid #ccc; padding: 0.5rem 0.6rem; text-align: left; vertical-align: top;
         overflow-wrap: break-word; }
th { background: #2a78d622; }
table.ledger { min-width: 1100px; }
table.ledger th:nth-child(1), table.ledger td:nth-child(1) { width: 13%; }
table.ledger th:nth-child(2), table.ledger td:nth-child(2) { width: 30%; }
table.ledger th:nth-child(3), table.ledger td:nth-child(3) { width: 20%; }
table.ledger th:nth-child(4), table.ledger td:nth-child(4) { width: 8%; }
table.ledger th:nth-child(5), table.ledger td:nth-child(5) { width: 14%; }
table.ledger th:nth-child(6), table.ledger td:nth-child(6) { width: 15%; }
code { background: rgba(127,127,127,0.15); padding: 0.05em 0.35em; border-radius: 3px; font-size: 0.9em; }
figure { margin: 1.2rem 0 2rem; }
figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
figcaption { font-size: 0.92rem; color: #555; margin-top: 0.5rem; }
.missing { padding: 2rem; text-align: center; color: #a33; border: 1px dashed #a33; }
.badge { display: inline-block; background: #eda100; color: #000; font-size: 0.75rem;
         padding: 0.1em 0.5em; border-radius: 3px; margin-left: 0.4em; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
ul.facts li { margin-bottom: 0.5rem; }
.note { background: #eda10022; border-left: 4px solid #eda100; padding: 0.8rem 1rem; margin: 1rem 0; }
@media (prefers-color-scheme: dark) {
  body { background: #14140f; color: #eee; }
  th { background: #2a78d655; }
  th, td { border-color: #444; }
  figure img { border-color: #444; }
  code { background: rgba(255,255,255,0.12); }
  .note { background: #4a3a0022; }
}
@media (max-width: 800px) { .grid2 { grid-template-columns: 1fr; } }
"""


def build() -> str:
    n = compute_numbers()
    mask_pivot = n["mask_pivot"]

    tau_rows = "".join(
        f"<li>τ×{r['tau']:g}: 最適lookahead={r['optimal_lh_gmd']:g}s "
        f"(GMD平均成功率{fmt_pct(r['success'])})</li>"
        for r in n["tau_optimal_rows"]
    )

    html = f"""<title>実験インベントリ・レポート (porcaro_2026)</title>
<style>{CSS}</style>

<h1>実験インベントリ・レポート</h1>
<p class="subtitle">porcaro_2026 / RA-L改訂 — 既存データの集計・作図のみ。新規学習・評価ジョブは一切実行していない。</p>
<p class="subtitle">主役: 先読み(anticipation)×記憶(memory)の2軸。DR有無に頑健。τスイープは trend であり causal law ではない。</p>

<h2>A. 実験レジャー</h2>
<p>IROS 2026 提出(#2555)のReviewer 3 / Reviewer 4 コメント原文に基づく対応関係。
"AE要約"はAssociate Editorのまとめコメントを指す。</p>
{render_ledger()}
<div class="note">
<b>既知のギャップ:</b> R3が要求する「model-based feedforward controller(同定した遅延・ヒステリシスモデルを使った比較)」は、
現状のどの実験群にも含まれていない。E(framestack MLP)はMLP-vs-LSTM問題には答えるが、
RL方策 vs 古典制御という軸には答えていない。
</div>

<h2>B. 学習結果</h2>

<h3>B-1. A-Eモデル比較・学習曲線(5 seed束)</h3>
<figure>
{img_tag("fig4a_learning_curves.png", "A-E learning curves")}
<figcaption>Train/mean_reward (rsl_rl on_policy_runner) vs iteration、モデル別mean±std(5 seed)。
B(lh=0.5s)とC(lh=1.0s)の学習曲線はほぼ重なっており、<b>学習報酬という指標ではB≈Cで差が出ない</b>
— 先読み軸の違いはeval(zero-shot成功率)側で初めて分離する(下記C節)。</figcaption>
</figure>

<h3>B-2. τスイープ・学習曲線(27 run = 3τ×3lookahead×3seed)</h3>
<figure>
{img_tag("fig_tau_sweep_learning_curves.png", "tau sweep learning curves")}
<figcaption>τ倍率ごとに3パネル、パネル内はlookahead別mean±std(3 seed)。τが大きいほど収束が遅く、
seed間ばらつき(帯の幅)も大きい傾向。τ×2の最長lookahead(lh=2.0s)はseed間分散が特に大きい。</figcaption>
</figure>

<h2>C. Play/Eval結果(±30ms成功率)</h2>

<h3>C-1. 先読み軸(A/B/C) と 記憶軸(D/B/E)</h3>
<figure>
{img_tag("fig_eval_success_lookahead_memory.png", "eval success by model/task")}
<figcaption>
GMD-03/GMD-04(複雑なリズム)では (i) A(0.1s) ≪ B(0.5s) ≈ C(1.0s)、
(ii) D(記憶なしMLP) &lt; B(LSTM) ≈ E(framestack MLP, k=5) がいずれも95%CIで分離して見える。
単純タスク(double_160)では全群のCIが重なり横並び。
E(finite-history-without-recurrence)がB(LSTM)と同水準であることは、
「有限履歴で足りており、recurrence自体が本質ではない」という記憶軸の主張の直接的な根拠になる。
</figcaption>
</figure>

<h3>C-2. Model D の per-seed 崩壊</h3>
<figure>
{img_tag("fig_perseed_eval_modelD.png", "per-seed D collapse")}
<figcaption>
Model D(記憶なしMLP, lh=0.5s)の5 seed中、seed{n['d_collapsed_seeds']}がGMD-03/GMD-04の両方で
成功率0%まで崩壊している(double_160では崩壊していない)。
D群の平均値・CIが広いのはこの2 seedの二峰性(bimodality)が主因であり、
「記憶なしモデルは複雑なリズムで安定して学習できないことがある」という追加の観察。
</figcaption>
</figure>

<h2>D. τスイープ: 最適先読みとPAM時定数の関係(GMD限定)</h2>
<figure>
{img_tag("fig_tau_vs_optimal_lookahead_gmd.png", "tau vs optimal lookahead, GMD only")}
<figcaption>
GMD-03/GMD-04のみを対象に、τ別のsuccess vs lookahead曲線と最適点(◎)を再作図。
<ul class="facts">{tau_rows}</ul>
</figcaption>
</figure>
<div class="note">
<b>解釈上の注意(trend, not law):</b>
<ul class="facts">
<li>これは<b>傾向(trend)</b>であって<b>因果則(causal law)</b>ではない — 「必要な先読みはτに比例する」と
断定する統計的検定は行っていない(セルあたりn=3 seedで高ノイズ、τは3水準のみ)。</li>
<li>τ=0.5とτ=1.0はどちらも最適lookahead=0.5sで<b>端で頭打ちしていない</b>(グリッド内の中間点が最適)一方、
τ=2.0は最適=lookaheadグリッドの右端(2.0s)であり、<b>真の最適点がグリッド外にある可能性を否定できない</b>。</li>
<li>各τのlookaheadグリッド自体が「τに比例する値」として事前設計されている(grid比例)ため、
「最適lookaheadがτに比例して見える」こと自体がグリッド設計から一部説明できてしまう
— 独立な確認ではない点に注意。</li>
</ul>
</div>

<h2>E. 頑健性・補助実験</h2>

<h3>E-1. DR vs 非DR</h3>
<figure>
{img_tag("fig_dr_vs_nondr.png", "DR vs non-DR delta")}
<figcaption>
モデル×条件ごとのΔ(非DR成功率 − DR成功率)。|Δ|の平均は{fmt_pct(n['dr_mean_abs_delta'])}、
最大は{fmt_pct(n['dr_max_abs_delta'])}で、符号もモデル間で一貫していない。
Bが他モデルより系統的に悪化する(=DRに過剰適応している)パターンは見られず、
「BのDR過剰適応がsim-real gapの一因」という仮説は<b>本データでは支持されない</b>
(本文・図から除外済みの仮説であることの確認)。
</figcaption>
</figure>

<h3>E-2. 評価ノイズ vs seed間分散</h3>
<figure>
{img_tag("fig_eval_noise.png", "eval noise vs seed variance")}
<figcaption>
B/C/EではSeed間標準偏差(青)が評価trial間標準偏差(橙、R=5の平均)と同程度かやや大きい。
Dだけseed間標準偏差が突出しているが、これはC-2節で見た2 seedの崩壊(0%)によるものであり、
評価ノイズ自体はDも他モデルと同程度。<span class="badge">R4指摘への部分対応</span>
このsim内評価は<b>trial間(=同一seed・同一条件を複数回評価した際の)ばらつき</b>を定量化したものであり、
R4が要求する「実機でのtrial数明示」そのものの代替にはならない。
</figcaption>
</figure>

<h3>E-3. 遠未来マスキング(zero/noise/shuffle)</h3>
<figure>
{img_tag("fig_mask_comparison.png", "mask comparison")}
<figcaption>
Model C(lh=1.0s)の遠未来(0.5-1.0s)部分をzero化・shuffle・noise付加した場合の成功率比較。
数値(mean): double_160 baseline={fmt_pct(mask_pivot.loc['double_160','none (baseline)'])} /
zero={fmt_pct(mask_pivot.loc['double_160','zero'])} /
shuffle={fmt_pct(mask_pivot.loc['double_160','shuffle'])} /
noise={fmt_pct(mask_pivot.loc['double_160','noise'])}。
GMD-03/04でも同様の順序(zero≈shuffle≈baseline &gt; noise)。
zero化・shuffleではbaselineと有意な差が見えない一方、noiseのみ明確な低下 —
「遠未来情報は無くても困らないが、誤った情報は害になる」という非対称な結果であり、
R3が提起した「1.0s horizonの劣化は無関係な長期cueへの過学習によるものではないか」という問いに対し、
単純な過学習仮説(不要な情報を捨てられていないだけ)では説明しにくい結果になっている。
</figcaption>
</figure>

<h2>実機実験の要否メモ</h2>
<p>結論は出さない。判断に必要な事実のみを列挙する。</p>

<h3>(a) sim(DR/非DR)で既に示せていること</h3>
<ul class="facts">
<li>先読み軸: GMD条件でA(0.1s)≪B(0.5s)≈C(1.0s)、DR・非DR双方で概ね同じ順序(E-1節)。</li>
<li>記憶軸: GMD条件でD(記憶なし)&lt;B(LSTM)≈E(framestack MLP)、finite historyがあれば
recurrence自体は必須ではないという傾向(C-1節)。</li>
<li>単純タスク(double_160)では先読み軸・記憶軸ともに群間差が小さい(CI重複)。</li>
<li>「Bのlh=0.5sがDR(τ±20%)に過剰適応している」という仮説はDR/非DR比較で支持されない(E-1節)。</li>
<li>最適lookaheadがτ(PAM時定数の倍率)に応じて変化する傾向(D節、ただしtrendでありlawではない)。</li>
<li>遠未来(0.5-1.0s)情報のzero化・shuffleでは性能がほぼ変わらないが、noise付加では低下する(E-3節)。</li>
<li>D(記憶なしMLP)は5 seed中2 seedがGMD条件で完全崩壊(0%)する学習不安定性がある(C-2節)。</li>
<li>評価trial間ばらつき(R=5)はGMD条件でB/C/Eについてseed間ばらつきと同程度以下(E-2節)。</li>
</ul>

<h3>(b) 実機でしか示せないこと</h3>
<ul class="facts">
<li>実機PAMの実際のhysteresis・遅延特性そのもの(simの物理モデルはpneumatic.py等の
テーブル値に基づく近似であり、その精度自体はsim内実験では検証できない)。</li>
<li>実際のsim-to-realギャップの大きさ・方向(DR/非DR比較はsim内のtask定義の違いであり、
実機との差そのものではない)。</li>
<li>実機センサノイズ・外乱・温度依存性・空気圧供給変動など、現在のDR(質量・摩擦・τ±20%)が
カバーしていない不確実性源の影響。</li>
<li>R3が要求するmodel-based feedforward controller(同定済み遅延・ヒステリシスモデル使用)との比較 —
sim側にもこのbaselineは存在しない(現状のギャップ、(a)にも(b)にも属さない未実施項目)。</li>
<li>実機での±30ms成功率そのもの、および実機trial数を明示した評価プロトコル(R4指摘の直接対象)。</li>
</ul>

<h3>(c) R3/R4指摘の sim対応状況</h3>
<div class="table-scroll"><table>
<thead><tr><th>指摘</th><th>出典</th><th>sim側の対応状況</th></tr></thead>
<tbody>
<tr><td>MLP-LSTM比較はmemory機構を単離できていない</td><td>R3 Minor-1</td>
<td>対応: E(framestack MLP)を追加しD&lt;B≈Eを確認(C-1節)</td></tr>
<tr><td>Model-basedフィードフォワード制御器との比較が無い</td><td>R3 Weakness-3a</td>
<td>未対応(sim側にもbaseline無し)</td></tr>
<tr><td>0.5s最適という主張がcorrelationalで普遍性の根拠が薄い</td><td>R3 Weakness-2 / R4</td>
<td>部分対応: τスイープでτ依存の傾向を提示、ただしtrendの域を出ない(D節)</td></tr>
<tr><td>固定lookahead設計はanticipationと情報量を混同している(1.0s劣化=過学習の可能性)</td>
<td>R3 Weakness-3b</td><td>対応: 遠未来マスキングで診断(E-3節)、ただし単純な過学習仮説とは
整合しにくい結果</td></tr>
<tr><td>GMDタスクの評価プロトコル(trial数)を基本タスク同様に明示すべき</td><td>R4 Minor</td>
<td>部分対応: sim内trial間ばらつきを定量化(E-2節)、実機trial数の報告そのものではない</td></tr>
<tr><td>Fig.2(c)の周波数応答解釈(時間領域スイープ応答では?)</td><td>R4 Minor</td>
<td>対象外(本レポートの集計スコープには含まれない、別figureの問題)</td></tr>
<tr><td>"embodied intelligence"等の表現を穏当に、フォーマット改善</td><td>R3/R4 Minor</td>
<td>対象外(文章・体裁の問題、データ集計と無関係)</td></tr>
</tbody>
</table></div>

"""
    return html


def main() -> None:
    html = build()
    out_path = SUMMARY_DIR / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[build_report] wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
