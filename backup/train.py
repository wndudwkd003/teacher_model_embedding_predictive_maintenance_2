# =========================================================
#  Tabular models: LightGBM · XGBoost · TabNet · FT-Transformer
#  with / without SMOTE on FD001+FD003 데이터만 사용
# =========================================================



import pandas as pd, numpy as np, torch, warnings, os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb, xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from tab_transformer_pytorch import FTTransformer, TabTransformer
warnings.filterwarnings('ignore')
torch.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ---------- 경로 ----------
CSV_IN      = Path('/dev/hdd/user/kjy/pi2/engine_knee_plots_multi/all_engines_labeled.csv')
METRIC_CSV  = Path('model_results.csv')
REPORT_DIR  = Path('reports'); REPORT_DIR.mkdir(exist_ok=True)

EPOCHS_TAB  = 200
EPOCHS_FT   = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")


# ---------- 데이터 ----------
df = pd.read_csv(CSV_IN)

# FD001, FD003만 사용
df = df[df['dataset'].isin(['FD001', 'FD003'])].copy()

drop_cols = [
    'unit', 'cycle', 'set1', 'set2', 'set3',
    's1', 's5', 's6', 's10', 's16', 's18', 's19',
    'state', 'dataset'
]

X = MinMaxScaler().fit_transform(df.drop(columns=drop_cols))
y = df['state'].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- 공통 함수 ----------
def save_report(tag, model, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    Path(REPORT_DIR / f'{tag}_{model}.txt').write_text(
        classification_report(y_true, y_pred, digits=3)
    )
    return acc, f1

def run_lgb(Xtr, ytr):
    print("[LightGBM] 학습 시작")
    m = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective='multiclass',
        random_state=42
    )
    m.fit(
        Xtr, ytr,
        eval_set=[(X_te, y_te)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )
    print("[LightGBM] 학습 완료")
    return m

def run_xgb(Xtr, ytr):
    print("[XGBoost] 학습 시작")
    m = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        use_label_encoder=False,
        early_stopping_rounds=20,
        random_state=42
    )
    m.fit(
        Xtr, ytr,
        eval_set=[(X_te, y_te)],
        verbose=50
    )
    print(f"[XGBoost] 학습 완료 - best_iteration = {m.best_iteration}")
    return m



def run_tabnet(Xtr, ytr):
    print("[TabNet] 학습 시작")
    m = TabNetClassifier(verbose=1, seed=42, device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(
        Xtr, ytr,
        eval_set=[(X_te, y_te)],
        max_epochs=EPOCHS_TAB,
        patience=20,  # 개선 없으면 20 epoch 후 중단
        eval_metric=['accuracy']
    )
    print("[TabNet] 학습 완료")
    return m


def run_ft(Xtr, ytr):
    print("[FTTransformer] 학습 시작")
    d = Xtr.shape[1]
    model = FTTransformer(
        categories=(),
        num_continuous=d,
        dim=64, depth=4, heads=8,
        dim_out=4, attn_dropout=0.1, ff_dropout=0.1
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(Xtr, dtype=torch.float32)
    y_t = torch.tensor(ytr, dtype=torch.long)
    X_val = torch.tensor(X_te, dtype=torch.float32)
    y_val = torch.tensor(y_te, dtype=torch.long)
    empty_cat = torch.empty((X_t.size(0), 0), dtype=torch.long)
    empty_val_cat = torch.empty((X_val.size(0), 0), dtype=torch.long)

    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(EPOCHS_FT):
        model.train(); opt.zero_grad()
        out = model(empty_cat, X_t)
        loss = torch.nn.functional.cross_entropy(out, y_t)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(empty_val_cat, X_val)
            val_loss = torch.nn.functional.cross_entropy(val_out, y_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[FTTransformer] Epoch {epoch+1}/{EPOCHS_FT} - TrainLoss: {loss.item():.4f}, ValLoss: {val_loss.item():.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[FTTransformer] Early stopping triggered at epoch {epoch+1}")
                break

    print("[FTTransformer] 학습 완료")
    return model


def predict_ft(model, Xv):
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    empty_cat = torch.empty((Xv_t.size(0), 0), dtype=torch.long)
    with torch.no_grad():
        return model(empty_cat, Xv_t).argmax(1).cpu().numpy()

def run_tabtransformer(Xtr, ytr):
    print("[TabTransformer] 학습 시작")
    d = Xtr.shape[1]
    mean_std = torch.tensor(
        np.stack([Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6], axis=1), dtype=torch.float32
    )
    model = TabTransformer(
        categories=(),  # 범주형 없음
        num_continuous=d,
        dim=64,
        dim_out=4,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=torch.nn.ReLU(),
        continuous_mean_std=mean_std
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(Xtr, dtype=torch.float32)
    y_t = torch.tensor(ytr, dtype=torch.long)
    X_val = torch.tensor(X_te, dtype=torch.float32)
    y_val = torch.tensor(y_te, dtype=torch.long)
    empty_cat = torch.empty((X_t.size(0), 0), dtype=torch.long)
    empty_val_cat = torch.empty((X_val.size(0), 0), dtype=torch.long)

    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(EPOCHS_FT):
        model.train(); opt.zero_grad()
        out = model(empty_cat, X_t)
        loss = torch.nn.functional.cross_entropy(out, y_t)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(empty_val_cat, X_val)
            val_loss = torch.nn.functional.cross_entropy(val_out, y_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[TabTransformer] Epoch {epoch+1}/{EPOCHS_FT} - TrainLoss: {loss.item():.4f}, ValLoss: {val_loss.item():.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[TabTransformer] Early stopping triggered at epoch {epoch+1}")
                break

    print("[TabTransformer] 학습 완료")
    return model


def predict_tabtransformer(model, Xv):
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    empty_cat = torch.empty((Xv_t.size(0), 0), dtype=torch.long)
    with torch.no_grad():
        return model(empty_cat, Xv_t).argmax(1).cpu().numpy()



def pipeline(tag, Xtrain, ytrain):
    print(f"\n========== [{tag.upper()}] 파이프라인 시작 ==========")
    results = []

    mdl = run_lgb(Xtrain, ytrain)
    pred = mdl.predict(X_te)
    acc, f1 = save_report(tag, 'lgb', y_te, pred)
    results.append((tag, 'LightGBM', acc, f1))

    mdl = run_xgb(Xtrain, ytrain)
    pred = mdl.predict(X_te)
    acc, f1 = save_report(tag, 'xgb', y_te, pred)
    results.append((tag, 'XGBoost', acc, f1))

    mdl = run_tabnet(Xtrain, ytrain)
    pred = mdl.predict(X_te)
    acc, f1 = save_report(tag, 'tabnet', y_te, pred)
    results.append((tag, 'TabNet', acc, f1))

    mdl = run_ft(Xtrain, ytrain)
    pred = predict_ft(mdl, X_te)
    acc, f1 = save_report(tag, 'ftt', y_te, pred)
    results.append((tag, 'FTTransformer', acc, f1))

    mdl = run_tabtransformer(Xtrain, ytrain)
    pred = predict_tabtransformer(mdl, X_te)
    acc, f1 = save_report(tag, 'tabtransformer', y_te, pred)
    results.append((tag, 'TabTransformer', acc, f1))

    print(f"========== [{tag.upper()}] 파이프라인 완료 ==========\n")
    return results

all_results = []
all_results += pipeline('no_smote', X_tr, y_tr)

X_aug, y_aug = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
all_results += pipeline('smote', X_aug, y_aug)

pd.DataFrame(all_results, columns=['setting', 'model', 'accuracy', 'macro_f1']).to_csv(METRIC_CSV, index=False)

print('완료!  metrics →', METRIC_CSV, ',  reports →', REPORT_DIR)
