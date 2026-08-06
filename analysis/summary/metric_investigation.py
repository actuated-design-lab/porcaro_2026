"""metric_investigation.py — 打点判定の3方式を同一データで比較する調査スクリプト。

2026-08-06。「実機のダブルストロークはもっと叩けていたはず」という指摘から出発して、
現行の打点判定にバグがあることを突き止めた記録。

  (i)  argmax-1.5T  現行。目標時刻の ±1.5*T16th の窓で force の argmax を取る。
                    → 目標間隔が窓より近いと、隣り合う2つの目標が同一の物理打撃を
                      掴む。160BPMのダブル（間隔80-120ms / 窓±140.6ms）で
                      実機47% / sim50% が該当し、成功率が0.5付近で頭打ちになる。
  (ii) argmax-mid   (i) の窓を隣接目標の中点で切ったもの。二重割当は消える。
  (iii) peak-match  IROS の create_fig7.py と同じ方式。find_peaks(force, >=1N) で
                    実打撃を検出し、|誤差| の小さい順に1対1で割り当てる。
                    **これが正しい実装。** マッチ窓を ±45〜±150ms で振っても
                    成功率は完全に不変（B 0.442 / C 0.508 / E 0.492）。

実機 double_160 の成功率:
        (i)      (ii)     (iii)
  B   0.275 -> 0.425 -> 0.442
  C   0.325 -> 0.458 -> 0.508
  E   0.350 -> 0.475 -> 0.492
  single8 は影響ゼロ（間隔498ms）、gmd_03 は +0.02 程度。

★ ただし指標を直しても 0.5 前後にとどまる。波形から直接数えると、ダブル1組の中に
  1N以上のピークが2つ出るのは B 50% / C 52% / E 47% しかない。残りは実体。
  失敗の内訳（B/C/E統合）: 打撃が無い 27.2% / ±30msを外れた 24.7% / 成功 48.1%。

★ 判定に力の大きさは入っていない。success = (peak_force >= 1N) and (|err| <= 30ms)
  で、力は二値ゲートのみ。ただし (i)(ii) は argmax なので「最も強いピーク」を選ぶ
  という隠れた力依存があり、これも二重割当の一因だった。(iii) は時間近接のみで選ぶ。
"""

import sys, os, glob, json, re
import numpy as np, pandas as pd
from scipy.signal import find_peaks
sys.path.insert(0,'/home/claude/research/jetson_project/analysis')
from strike_metrics import apply_time_source

TOL=30.0; THR=1.0

def targets(t, tgt, t16):
    chg=np.r_[True, np.diff(tgt)!=0.0]; tc,gc=t[chg],tgt[chg]
    dt=float(np.median(np.diff(tc))) if len(tc)>1 else 0.02
    idx,_=find_peaks(gc, height=5.0, distance=max(1,int(round(0.3*t16/dt))))
    return tc[idx]

def m_argmax(t,F,tt,t16,clip):
    out=[]
    for i,x in enumerate(tt):
        lo,hi=x-1.5*t16,x+1.5*t16
        if clip:
            if i>0: lo=max(lo,(tt[i-1]+x)/2)
            if i<len(tt)-1: hi=min(hi,(x+tt[i+1])/2)
        m=(t>=lo)&(t<=hi)
        if not m.any(): out.append((np.nan,0.0)); continue
        lt,lf=t[m],F[m]; j=int(np.argmax(lf))
        out.append(((lt[j]-x)*1000.0, float(lf[j])))
    return out

def m_peak(t,F,tt,t16,match_win=0.150):
    """実打撃をピーク検出し、|誤差|の小さい順に1対1で割り当てる。"""
    dt=float(np.median(np.diff(t)))
    pk,_=find_peaks(F, height=THR, distance=max(1,int(round(0.3*t16/dt))))
    pt,pf=t[pk],F[pk]
    cand=[]
    for i,x in enumerate(tt):
        for j in range(len(pt)):
            e=(pt[j]-x)*1000.0
            if abs(e)<=match_win*1000: cand.append((abs(e),i,j,e,pf[j]))
    cand.sort()
    used_t,used_p={},set()
    for _,i,j,e,f in cand:
        if i in used_t or j in used_p: continue
        used_t[i]=(e,f); used_p.add(j)
    return [used_t.get(i,(np.nan,0.0)) for i in range(len(tt))]

rows=[]
for p in sorted(glob.glob('/home/claude/research/jetson_project/results/RAL/*/*.csv')):
    jp=os.path.splitext(p)[0]+'.json'
    if not os.path.exists(jp): continue
    meta=json.load(open(jp))
    if meta.get('mock') or meta.get('verify'): continue
    df=pd.read_csv(p)
    if 'target_force' not in df.columns: continue
    bpm=float(meta.get('bpm',0) or 0)
    if bpm<=0: continue
    df=apply_time_source(df,'rate_corrected')
    t=df.time.to_numpy(float); tgt=df.target_force.to_numpy(float); F=df.force_N.to_numpy(float)
    t16=15.0/bpm; tt=targets(t,tgt,t16)
    if len(tt)==0: continue
    mk=str(meta.get('model_key','')); mm=re.search(r'/([A-E])_seed',mk)
    if not mm: continue
    base=dict(model=mm.group(1), seed=int(re.search(r'seed(\d)',mk).group(1)),
              midi=os.path.basename(meta.get('midi','')), trial=meta.get('trial',1))
    for name,res in [("(i) argmax-1.5T", m_argmax(t,F,tt,t16,False)),
                     ("(ii) argmax-mid", m_argmax(t,F,tt,t16,True)),
                     ("(iii) peak-match", m_peak(t,F,tt,t16))]:
        succ=np.mean([ (abs(e)<=TOL and f>=THR) if e==e else False for e,f in res])
        struck=np.mean([f>=THR for e,f in res])
        rows.append({**base, 'method':name, 'success':succ, 'struck':struck, 'n':len(tt)})
r=pd.DataFrame(rows); r.to_csv('/tmp/metric3.csv',index=False)
for midi in ['test_single8_bpm120.mid','test_double_bpm160.mid','gmd_03_high_bpm138.mid']:
    sub=r[r.midi==midi]
    piv=sub.groupby(['method','model','seed']).success.mean().groupby(['method','model']).mean().unstack(0)
    print(f"\n=== {midi}（実機・シード単位の成功率）===")
    print(piv[["(i) argmax-1.5T","(ii) argmax-mid","(iii) peak-match"]].round(3).to_string())
