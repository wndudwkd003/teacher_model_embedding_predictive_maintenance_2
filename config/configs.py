from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch


class FrameType(Enum):
    RAW_RAW = "raw_raw"
    RAW_LOGITS = "raw_logits"
    EMBEDDING_RAW = "embedding_raw"
    EMBEDDING_LOGITS = "embedding_logits"
    EMBEDDING_MIX_RAW = "embedding_mix_raw"


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


class RunType(Enum):
    TRAIN = "train"
    TEST = "test"
    TRAIN_TEST = "train_test"


class StackType(Enum):
    STACK = "stack"
    NO_STACK = "no_stack"



@dataclass
class Config:
    # 학습 또는 평가 모드 설정하는 변수
    run_mode: RunType = RunType.TRAIN_TEST

    # 학습 하이퍼파라미터 변수
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size: int = 512
    num_epochs: int = 100000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    scheduler_patience: int = 5
    seed: int = 42
    dropout: float = 0.2


    # 시나리오 수
    train_scenario: List[int] = field(default_factory=lambda: list(range(14)))

    # stack type
    train_data_stack_type: StackType = StackType.NO_STACK
    test_data_stack_type: StackType = StackType.NO_STACK

    # train path
    train_data_path: str = f"dataset_output_for_train_{train_data_stack_type.value}"
    train_valid_result_path: str = f"outputs/train_result_{train_data_stack_type.value}"

    # test path
    test_data_path: str = f"dataset_output_for_masking_test_{test_data_stack_type.value}"
    test_result_path: str = f"outputs/test_results_{test_data_stack_type.value}"
    test_model_save_path: str = "outputs/train_result_no_stack/RAW_RAW_TabNet_20250518_165155"


    # 학습 방법 설정
    is_kd_mode: bool = False # False -> 무조건 student 모델의 설정대로 진행
    embedding_type: DataType = DataType.BERT
    teacher_frame_type: FrameType = FrameType.RAW_RAW
    teacher_model_type: ModelType = ModelType.XGBOOST
    student_frame_type: FrameType = FrameType.EMBEDDING_MIX_RAW
    student_model_type: ModelType = ModelType.MLP

    # XGBoost
    early_stopping_rounds: int = 10
    xgb_device: str = "gpu" if torch.cuda.is_available() else "cpu"
    eval_metric: str = "mlogloss"
    tree_method: str = "hist"
    max_depth: int = 6
    subsample: float = 0.9

    # 딥러닝 설정
    raw_input_dim: int = 14
    embedding_input_dim: int = 768
    mix_input_dim: int = embedding_input_dim+raw_input_dim

