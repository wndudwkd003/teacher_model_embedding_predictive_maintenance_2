import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ---------- 경로 ----------
CSV_PATH = 'model_results.csv'
OUT_DIR = Path('plots'); OUT_DIR.mkdir(exist_ok=True)

# ---------- 색상 설정 ----------
color_acc = 'skyblue'
color_f1  = 'salmon'

# ---------- 데이터 로드 ----------
df = pd.read_csv(CSV_PATH)
settings = df['setting'].unique()

# ---------- 시각화 ----------
for setting in settings:
    sub_df = df[df['setting'] == setting].copy()
    models = sub_df['model'].tolist()
    acc = sub_df['accuracy'].values
    f1  = sub_df['macro_f1'].values

    y = np.arange(len(models))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y - bar_width/2, acc, height=bar_width, color=color_acc, label='Accuracy')
    ax.barh(y + bar_width/2, f1,  height=bar_width, color=color_f1,  label='Macro F1')

    # 수치 표시
    for i in range(len(models)):
        ax.text(acc[i] + 0.01, y[i] - bar_width/2, f'{acc[i]:.3f}', va='center', fontsize=9)
        ax.text(f1[i] + 0.01, y[i] + bar_width/2, f'{f1[i]:.3f}', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Score')
    ax.set_title(f'Model Performance ({setting})')
    ax.legend(loc='lower right')
    plt.tight_layout()

    save_path = OUT_DIR / f'{setting}_horizontal_bar.png'
    plt.savefig(save_path)
    print(f'✅ 저장 완료: {save_path}')
