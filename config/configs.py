from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch


class FrameType(Enum):
    RAW_RAW = "raw_raw"
    RAW_LOGITS = "raw_logits"
    EMBEDDING_RAW = "embedding_raw"
    EMBEDDING_LOGITS = "embedding_logits"


class ModelType(Enum):
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    MLP = "MLP"
    RESNET_MLP = "ResidualMLP"
    TABNET = "TabNet"


class DataType(Enum):
    RAW = "RAW"
    GPT2 = "GPT2"
    BERT = "BERT"


@dataclass
class Config:
    batch_size: int = 512
    num_epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    scheduler_patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    dropout: float = 0.2

    train_model_save_path: str = "model_save"
    train_valid_result_path: str = "output_result"

    train_scenario: List[int] = field(default_factory=lambda: list(range(14)))

    data_path: str = "dataset_output_for_train"

    embedding_type: DataType = DataType.RAW

    is_kd_mode: bool = False # False -> 무조건 student 모델의 설정대로 진행
    teacher_data_type: FrameType = FrameType.RAW_RAW
    teacher_model_type: ModelType = ModelType.XGBOOST
    student_data_type: FrameType = FrameType.RAW_RAW
    student_model_type: ModelType = ModelType.MLP

    mlp_input_dim: int = 14 if student_data_type == FrameType.RAW_RAW else 768


    # XGBoost
    early_stopping_rounds: int = 10
    xgb_device: str = "gpu" if torch.cuda.is_available() else "cpu"
    eval_metric: str = "mlogloss"
    tree_method: str = "hist"
    max_depth: int = 6
    subsample: float = 0.9


