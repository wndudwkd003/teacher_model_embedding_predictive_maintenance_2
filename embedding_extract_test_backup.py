"""
# 언어 모델의 임베딩을 추출하는 코드
# 학습용

"""

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb, xgboost as xgb
from imblearn.over_sampling import SMOTE
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
    models: List[ModelType] = field(default_factory=lambda: [ModelType.BERT, ModelType.GPT2, ModelType.GEMMA3, ModelType.T5, ModelType.ELECTRA])
    dataset_select: List[str] = field(default_factory=lambda: ["FD001", "FD003"])
    dataset_path: str = "/home/juyoung-lab/ws/dev_ws/pi2/engine_knee_plots_multi_no_normal/all_engines_labeled.csv"
    type: str = "train" # train or masking_test 
    drop_cols: List[str] = field(default_factory=lambda: ['unit','cycle','set1','set2','set3', 's1','s5','s6','s10','s16','s18','s19','state','dataset'])
    cols_rename_map: Dict[str, str] = field(default_factory=lambda: {
            "unit": "engine_id",
            "cycle": "flight_cycle",
            "set1": "altitude",         # 고도
            "set2": "mach_number",      # 마하 수
            "set3": "throttle_resolver_angle",  # 스로틀 위치
            "s1": "fan_inlet_temp_T2",
            "s2": "LPC_outlet_temp_T24",
            "s3": "HPC_outlet_temp_T30",
            "s4": "LPT_outlet_temp_T50",
            "s5": "fan_inlet_pressure_P2",
            "s6": "bypass_duct_pressure_P15",
            "s7": "HPC_outlet_pressure_P30",
            "s8": "fan_speed_Nf",
            "s9": "core_speed_Nc",
            "s10": "engine_pressure_ratio_epr",
            "s11": "HPC_static_pressure_Ps30",
            "s12": "fuel_flow_ratio_phi",
            "s13": "corrected_fan_speed_NRf",
            "s14": "corrected_core_speed_NRc",
            "s15": "bypass_ratio_BPR",
            "s16": "burner_fuel_air_ratio_farB",
            "s17": "bleed_enthalpy_htBleed",
            "s18": "demanded_fan_speed_Nf_dmd",
            "s19": "demanded_corrected_fan_speed_PCNfR_dmd",
            "s20": "HPT_coolant_bleed_W31",
            "s21": "LPT_coolant_bleed_W32",
            "state": "label",
            "dataset": "dataset_id"
        })

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def printi(msg: str):
    print(f"[INFO] {msg}")


def split_dataset(df: pd.DataFrame, dataset_select: List[str], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_df = df[df["dataset"].isin(dataset_select)].copy()
    groups = selected_df["unit"]  # 'unit'은 나중에 'engine_id'로 이름 바뀌기 전 기준
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(selected_df, groups=groups))
    train_df = selected_df.iloc[train_idx].reset_index(drop=True)
    test_df = selected_df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

# random split 
# def split_dataset(df: pd.DataFrame, dataset_select: List[str], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     selected_df = df[df["dataset"].isin(dataset_select)].copy()
#     train_df, test_df = train_test_split(selected_df, test_size=0.2, random_state=seed, stratify=selected_df["state"])
#     return train_df, test_df


def main(config: Config):
    # 언어 모델 임베딩 추출
    
    device = config.device if torch.cuda.is_available() else "cpu"
    printi(f"Using device: {device}")

    df = pd.read_csv(config.dataset_path)
    train_df, test_df = split_dataset(df, config.dataset_select, config.seed)

    X_train, y_train = train_df.drop(columns=config.drop_cols), train_df["state"].values
    X_test, y_test = test_df.drop(columns=config.drop_cols), test_df["state"].values

    X_train.rename(columns=config.cols_rename_map, inplace=True)
    X_test.rename(columns=config.cols_rename_map, inplace=True)

    X_train = X_train.values
    X_test = X_test.values


    original_counts = Counter(y_train)
    printi(f"Original counts: {original_counts}")
    total_target = int(len(y_train) * 5)
    scaling = total_target / len(y_train)
    sampling_strategy = {k: int(v * scaling) for k, v in original_counts.items()}

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=config.seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    printi(f"Resampled counts: {Counter(y_train)}")

    printi(f"Train: {X_train.shape}, {y_train.shape} | Test: {X_test.shape}, {y_test.shape}")


    def save_report(tag, y_true, y_pred):
        report = classification_report(y_true, y_pred, digits=3, output_dict=False)
        print(tag)
        print(report)
        print()

    print("[LightGBM] 학습 시작")

    m = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective='multiclass',
        random_state=42
    )
    m.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )

    save_report("lgbm", y_test, m.predict(X_test))

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
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    save_report("xgboost", y_test, m.predict(X_test))



    print("[TabNet] 학습 시작")

    m = TabNetClassifier(verbose=1, seed=42, device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=1000,
        patience=20,  
        eval_metric=['accuracy']
    )
    pred = m.predict(X_test)
    save_report("tabnet", y_test, pred)


    # 학습용으로 임베딩 추출 -> 시나리오 0부터 센서 개수(16-1)까지 랜덤으로 결측 마스킹
    if config.type == "train":
        pass 
    

    # 테스트용으로 임베딩 추출 -> lgbm feature importance로 시나리오 순서대로 결측 마스킹
    else:
        pass 
        
    

    


if __name__ == "__main__":
    config = Config()
    seed_everything(config.seed)
    main(config)