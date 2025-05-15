import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from collections import Counter


class ModelType(Enum):
    BERT = "bert-base-uncased"
    GPT2 = "gpt2"
    GEMMA3 = "google/gemma-3-1b-pt"
    T5 = "t5-base"
    ELECTRA = "google/electra-base-discriminator"

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    dataset_select: List[str] = field(default_factory=lambda: ["FD001", "FD003"])
    dataset_path: str = "engine_knee_plots_multi/all_engines_labeled.csv"
    drop_cols: List[str] = field(default_factory=lambda: [
        'unit', 'cycle', 'set1', 'set2', 'set3', 's1', 's5', 's6', 's10', 's16', 's18', 's19', 'state', 'dataset'])
    cols_rename_map: Dict[str, str] = field(default_factory=lambda: {
        "unit": "engine_id", "cycle": "flight_cycle", "set1": "altitude", "set2": "mach_number",
        "set3": "throttle_resolver_angle", "s1": "fan_inlet_temp_T2", "s2": "LPC_outlet_temp_T24",
        "s3": "HPC_outlet_temp_T30", "s4": "LPT_outlet_temp_T50", "s5": "fan_inlet_pressure_P2",
        "s6": "bypass_duct_pressure_P15", "s7": "HPC_outlet_pressure_P30", "s8": "fan_speed_Nf",
        "s9": "core_speed_Nc", "s10": "engine_pressure_ratio_epr", "s11": "HPC_static_pressure_Ps30",
        "s12": "fuel_flow_ratio_phi", "s13": "corrected_fan_speed_NRf", "s14": "corrected_core_speed_NRc",
        "s15": "bypass_ratio_BPR", "s16": "burner_fuel_air_ratio_farB", "s17": "bleed_enthalpy_htBleed",
        "s18": "demanded_fan_speed_Nf_dmd", "s19": "demanded_corrected_fan_speed_PCNfR_dmd",
        "s20": "HPT_coolant_bleed_W31", "s21": "LPT_coolant_bleed_W32", "state": "label", "dataset": "dataset_id"
    })


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def printi(msg):
    print(f"[INFO] {msg}")


def split_dataset(df: pd.DataFrame, dataset_select: List[str], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_df = df[df["dataset"].isin(dataset_select)].copy()
    groups = selected_df["unit"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(selected_df, groups=groups))
    return selected_df.iloc[train_idx].reset_index(drop=True), selected_df.iloc[test_idx].reset_index(drop=True)


def safe_resample(sampler, X, y):
    """sampler 실패 시 빈 배열 반환"""
    try:
        Xr, yr = sampler.fit_resample(X, y)
        return Xr, yr
    except ValueError:
        return X, y          # 증강 못 하면 원본 그대로

def apply_augmentation(X, y, method: str, multiplier: int, seed: int):
    if multiplier == 1:             # 배수 1 → 증강 없음
        if method == "MIX":
            return X, y             # MIX ×1 도 원본 그대로
        if method in {"SMOTE", "ADASYN", "BorderlineSMOTE"}:
            return X, y

    # =============== 개별 기법 ===============
    if method in {"SMOTE", "ADASYN", "BorderlineSMOTE"}:
        strat = {cls: max(int(cnt * multiplier), cnt + 1) for cls, cnt in Counter(y).items()}
        sampler = {
            "SMOTE": SMOTE(random_state=seed, sampling_strategy=strat),
            "ADASYN": ADASYN(random_state=seed, sampling_strategy=strat),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=seed, sampling_strategy=strat)
        }[method]
        Xr, yr = safe_resample(sampler, X, y)
        return Xr, yr

    # =============== MIX ===============
    # multiplier > 1
    X_sm, y_sm = safe_resample(SMOTE(random_state=seed), X, y)
    X_ad, y_ad = safe_resample(ADASYN(random_state=seed), X, y)
    X_bl, y_bl = safe_resample(BorderlineSMOTE(random_state=seed), X, y)

    # 새로 생긴 부분만 추출
    base = len(y)
    X_new = np.concatenate([X_sm[base:], X_ad[base:], X_bl[base:]])
    y_new = np.concatenate([y_sm[base:], y_ad[base:], y_bl[base:]])

    # (multiplier − 1) 배만큼 반복해서 붙이기
    if len(X_new):
        X_new = np.tile(X_new, (multiplier - 1, 1))
        y_new = np.tile(y_new, multiplier - 1)
        X_aug = np.concatenate([X, X_new])
        y_aug = np.concatenate([y, y_new])
        return X_aug, y_aug
    else:
        # 아무 증강도 못 했으면 원본 반환
        return X, y


def evaluate_models(X_train, y_train, X_test, y_test):
    results = {}

    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
    lgbm.fit(X_train, y_train)
    pred = lgbm.predict(X_test)
    results["LightGBM"] = (accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro'))

    xgbm = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                             objective='multi:softprob', num_class=len(set(y_train)),
                             use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgbm.fit(X_train, y_train)
    pred = xgbm.predict(X_test)
    results["XGBoost"] = (accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro'))

    tabnet = TabNetClassifier(seed=42, verbose=0)
    tabnet.fit(X_train=X_train, y_train=y_train, eval_set=[(X_test, y_test)],
               eval_metric=['accuracy'], patience=20, max_epochs=200)
    pred = tabnet.predict(X_test)
    results["TabNet"] = (accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro'))

    return results


def save_results_to_csv_and_plot(results):
    df = pd.DataFrame(results)
    df.to_csv("augmentation_results.csv", index=False)

    plt.figure(figsize=(14, 8))
    for method in df['method'].unique():
        subset = df[(df['model'] == 'LightGBM') & (df['method'] == method)]
        plt.plot(subset['multiplier'], subset['f1_macro'], marker='o', label=method)
    plt.title("LightGBM - Macro F1 vs Augmentation Multiplier")
    plt.xlabel("Multiplier")
    plt.ylabel("Macro F1 Score")
    plt.grid(True)
    plt.legend()
    plt.savefig("augmentation_f1_lightgbm.png")
    plt.close()





def main(config: Config):
    seed_everything(config.seed)

    df = pd.read_csv(config.dataset_path)
    train_df, test_df = split_dataset(df, config.dataset_select, config.seed)

    X_train_df = train_df.drop(columns=config.drop_cols)
    y_train = train_df["state"].values
    X_train_df.rename(columns=config.cols_rename_map, inplace=True)

    X_test_df = test_df.drop(columns=config.drop_cols)
    y_test = test_df["state"].values
    X_test_df.rename(columns=config.cols_rename_map, inplace=True)

    X_train = X_train_df.values
    X_test = X_test_df.values

    methods = ["SMOTE", "ADASYN", "BorderlineSMOTE", "MIX"]
    multipliers = [1, 3, 5, 10]
    all_results = []

    for method in methods:
        for mult in multipliers:
            printi(f"▶ Running: {method} x{mult}")
            X_aug, y_aug = apply_augmentation(X_train, y_train, method, mult, config.seed)
            result = evaluate_models(X_aug, y_aug, X_test, y_test)
            for model, (acc, f1) in result.items():
                all_results.append({
                    "method": method,
                    "multiplier": mult,
                    "model": model,
                    "accuracy": acc,
                    "f1_macro": f1
                })

    save_results_to_csv_and_plot(all_results)


if __name__ == "__main__":
    config = Config()
    main(config)
