import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from config.configs import Config, DataType, ModelType, FrameType
from models.ml_model_trainers import XGBoostTrainer
from models.factories import TrainerFactory
from models.data_module import DataFactory


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

SAVE_PATH = None


def main(config: Config):
    """
    # 계획
    1. 딥러닝, 머신러닝 비교 모델로 전부 학습해야 함
    2. 시나리오별로 학습하고 평가도 시나리오별로 진행
    3. 잘된 시나리오 모델을 최종 선정
    4. 평가 기준 적분 이용하는 방법 구상
    5. 교사-학생 모델 조합 각각 진행
    6. 한번에 학습 하는 구조가 아니라 Config에서 선택할 수 있도록 구성

    """
    global SAVE_PATH
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_valid_result_path = config.train_valid_result_path
    train_valid_result_path = os.path.join(train_valid_result_path, f"{config.student_data_type.name}_{config.student_model_type.value}_{current_time}")
    os.makedirs(train_valid_result_path, exist_ok=True)
    SAVE_PATH = train_valid_result_path

    train_scen = config.train_scenario
    reports = []

    # train by scenario
    for scen in train_scen:
        print(f"[INFO] Current Scenario: {scen}")
        report = train_pipe(scen, config)
        report["scenario"] = scen
        print("accuracy: ", report["accuracy"])
        reports.append(report)

    # valid report save
    report_df = pd.DataFrame(reports)
    report_df.to_csv(os.path.join(SAVE_PATH, "report.csv"), index=False)

    # valid

def data_load(scen: int, data_path: str, data_type: DataType):
    data_path = os.path.join(data_path, data_type.name, f"iteration_{scen}")
    if data_type == DataType.RAW:
        X_train = pd.read_csv((os.path.join(data_path, "X_train.csv"))).values
        y_train = pd.read_csv((os.path.join(data_path, "y_train.csv"))).values
        X_valid = pd.read_csv((os.path.join(data_path, "X_valid.csv"))).values
        y_valid = pd.read_csv((os.path.join(data_path, "y_valid.csv"))).values
        return [X_train, y_train, X_valid, y_valid]
    else:
        X_train = np.load(os.path.join(data_path, "X_train.npy"))
        X_valid = np.load(os.path.join(data_path, "X_valid.npy"))
        return [X_train, None, X_valid, None]


def train_pipe(scen: int, config: Config):
    global SAVE_PATH

    # data laod todo
    X_train, y_train = DataFactory.load(config.data_path, "train", scen, DataType.RAW)
    X_valid, y_valid = DataFactory.load(config.data_path, "valid", scen, DataType.RAW)

    if config.student_data_type != DataType.RAW:
        X_train, _ = DataFactory.load(config.data_path, "train", scen, config.embedding_type)
        X_valid, _ = DataFactory.load(config.data_path, "valid", scen, config.embedding_type)

    # model trainer
    trainer = TrainerFactory.build(config.student_model_type, config)
    trainer.fit(X_train, y_train, X_valid, y_valid, verbose=False)
    report = trainer.eval(X_valid, y_valid, digits=5, output_dict=True)

    # save path
    model_save_path = os.path.join(SAVE_PATH, config.train_model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

    # model save
    trainer.save_model(model_save_path, f"scen_{scen}")

    return report


if __name__ == "__main__":
    config = Config()
    seed_everything(config.seed)
    main(config)
