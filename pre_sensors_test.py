import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

CSV_PATH = Path('/dev/hdd/user/kjy/pi2/engine_knee_plots_multi/all_engines_labeled.csv')
OUT_DIR  = Path('/dev/hdd/user/kjy/pi2/engine_knee_plots_multi/viz')
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_COLORS = ['#8fd175', '#fff07e', '#f6b08c', '#d9534f']
DROP         = ['s1','s5','s6','s10','s16','s18','s19']
COLS         = ['s'+str(i) for i in range(1,22) if f's{i}' not in DROP]

df_all = pd.read_csv(CSV_PATH)

for fd in df_all['dataset'].unique():
    df_fd = df_all[df_all['dataset'] == fd]
    for eid in df_fd['unit'].unique():
        df_e = df_fd[df_fd['unit'] == eid].sort_values('cycle')
        cyc = df_e['cycle'].to_numpy()
        state = df_e['state'].to_numpy()

        fig, ax = plt.subplots(figsize=(14, 4))

        # 센서별로 구간 나눠서 색 다르게 그리기
        for sensor in COLS:
            if sensor not in df_e.columns: continue
            values = df_e[sensor].to_numpy()
            # state가 바뀌는 지점 인덱스
            change_idx = np.where(np.diff(state) != 0)[0] + 1
            seg_idx = [0] + change_idx.tolist() + [len(state)]

            for i in range(len(seg_idx) - 1):
                s, e = seg_idx[i], seg_idx[i+1]
                c = STATE_COLORS[state[s]] if state[s] < len(STATE_COLORS) else 'black'
                ax.plot(cyc[s:e], values[s:e], color=c, lw=0.7, alpha=0.8)

        ax.set_title(f'{fd} – Engine {eid} (per-state colored)')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Sensor Value (Normalized)')
        fig.tight_layout()
        fig.savefig(OUT_DIR / f'{fd}_engine_{eid}.png', dpi=150)
        plt.close(fig)

print('✔ 색상 적용된 선 그래프 저장 완료 →', OUT_DIR)
