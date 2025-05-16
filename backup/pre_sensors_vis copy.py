import pandas as pd, json, sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np

# ── 설정 ────────────────────────────────────────────
DATASETS  = ['FD001', 'FD002', 'FD003', 'FD004']   # 처리할 파일
SHIFT     = 0                                      # 0 = 그대로, 1·2 … = 뒤로 밀기
BASE_DIR  = Path('archive/CMaps')
MAP_FILE  = Path('sensor_udc_map.json')
OUT_ROOT  = Path('engine_knee_plots_multi_no_normal')
OUT_ROOT.mkdir(exist_ok=True)

# ── 매핑 로드 ───────────────────────────────────────
if not MAP_FILE.exists():
    sys.exit('sensor_udc_map.json not found')
mapping_all = json.loads(MAP_FILE.read_text())      # {FDxxx: {sensor: tag}}

# ── 공통 정보 ──────────────────────────────────────
DROP   = ['s1','s5','s6','s10','s16','s18','s19']
COLS   = ['unit','cycle','set1','set2','set3'] + [f's{i}' for i in range(1,22)]
COLORS = ['#8fd175', '#fff07e', '#f6b08c', '#d9534f']  # normal→danger
ALPHA  = 0.15

def edges_10(y, tag):
    x = np.arange(len(y))
    if   tag == 'u':  y1, curve, d = y, 'concave', 'increasing'
    elif tag == 'd':  y1, curve, d = y, 'convex',  'decreasing'
    else:             y1, curve, d = np.abs(y-y.mean()), 'concave', 'increasing'

    k = KneeLocator(x, y1, curve=curve, direction=d, S=2.0)
    idx = sorted(k.all_knees)[:9]
    while len(idx) < 9:
        q = int(len(y) * (len(idx)+1) / 10)
        if q not in idx: idx.append(q)
    idx = sorted(idx)[:9]
    return [0] + idx + [len(y)-1]                   # 11 경계

def smooth(v, alpha=0.05):
    return pd.Series(v).ewm(alpha=alpha, adjust=False).mean().to_numpy()

all_labeled = []  # ← 전체 결과를 모을 리스트

# ── 데이터셋 루프 ──────────────────────────────────
for fd in DATASETS:
    print(f'\n=== Processing {fd} ===')
    fd_map = mapping_all.get(fd)
    if not fd_map:
        print('  ↳ 매핑이 없습니다. 건너뜀');  continue

    fpath = BASE_DIR / f'train_{fd}.txt'
    df_raw = pd.read_csv(fpath, sep=r'\s+', header=None, names=COLS)

    sensors = [s for s in df_raw.columns if s.startswith('s') and s[1:].isdigit() and s not in DROP and s in fd_map]
    df = df_raw.copy()
    # df_norm = df_raw.copy()
    for s in sensors:
        df[s] = MinMaxScaler().fit_transform(df[[s]])

    groups = {k: [s for s in sensors if fd_map[s] == k] for k in ['u','d','c','o']}
    out_dir = OUT_ROOT / fd; out_dir.mkdir(exist_ok=True)

    for eid, g in df.groupby('unit'):
        g = g.sort_values('cycle')
        cyc = g.cycle.to_numpy()
        fig, ax = plt.subplots(figsize=(14, 4))

        for s in sensors:
            ax.plot(cyc, g[s], color='grey', alpha=.3, lw=.35)

        for tag, cols in groups.items():
            if not cols: continue
            m_raw  = g[cols].mean(axis=1).values
            m_line = smooth(m_raw)

            edges = edges_10(m_raw, tag)
            base  = [min(i, 10) for i in [SHIFT, SHIFT+3, SHIFT+6, SHIFT+9, 10]]
            seg_idx = [edges[i] for i in base]
            seg_cyc = [cyc[i] for i in seg_idx]; seg_cyc[-1] += 1

            for (l,r),c in zip(zip(seg_cyc[:-1], seg_cyc[1:]), COLORS):
                ax.axvspan(l, r, color=c, alpha=ALPHA)

            ax.plot(cyc, m_line, lw=2, label=f'{tag} mean')

            # ── 상태 라벨링 추가 ──
            state_label = np.zeros(len(cyc), dtype=int)
            for i in range(len(seg_cyc)-1):
                mask = (cyc >= seg_cyc[i]) & (cyc < seg_cyc[i+1])
                state_label[mask] = i

            g_out = g.copy()
            g_out['state'] = state_label
            g_out['dataset'] = fd
            all_labeled.append(g_out)  # 통합 리스트에 추가

        ax.set_xlabel('Cycle'); ax.set_ylabel('Scaled Value')
        ax.set_title(f'{fd} – Engine {eid}  (shift={SHIFT})')
        ax.legend(loc='upper left', fontsize='small')
        fig.tight_layout()
        fig.savefig(out_dir / f'{fd}_engine_{eid}.png', dpi=150)
        plt.close(fig)
        print(f'  ↳ {fd} engine {eid} → {out_dir / f"{fd}_engine_{eid}.png"}')

# ── 통합 CSV 저장 ─────────────────────────────────
df_all = pd.concat(all_labeled, ignore_index=True)
df_all.to_csv(OUT_ROOT / 'all_engines_labeled.csv', index=False)
print('\n✔ 통합 CSV 저장 완료 →', OUT_ROOT / 'all_engines_labeled.csv')