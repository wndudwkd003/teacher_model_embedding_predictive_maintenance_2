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
    EMBEDDING_MIX_LOGITS = "embedding_mix_logits"


class ModelType(Enum):
    MLP = "MLP"
    RESNET_MLP = "ResidualMLP"

    AE = "AutoEncoder"
    VAE = "VAE"
    AE_LATENT = "AautoEncoder_Latent"
    VAE_LATENT = "VAE_Latent"
    GNN = "GNN"

    TABNET = "TabNet"
    TAB_TRANSFORMER = "TabTransformer"
    FTT = "FTTransformer"

    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    SVM = "SVM"
    RF = "RandomForest"
    GBT = "GradientBoosting"
    LR = "LogisticRegression"


class DataType(Enum):
    RAW = "RAW"
    GPT2 = "GPT2"
    BERT = "BERT"
    GEMMA3 = "GEMMA3"
    ELECTRA = "ELECTRA"
    T5 = "T5"


class RunType(Enum):
    TRAIN = "train"
    TEST = "test"
    TRAIN_TEST = "train_test"


class StackType(Enum):
    STACK = "stack"
    NO_STACK = "no_stack"
    WITH_SMOTE_NO_STACK = "with_smote_no_stack"



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

    # 실제 학습 및 검증 시 데이터 설정
    train_data_stack_type: StackType = StackType.STACK
    test_data_stack_type: StackType = StackType.STACK

    # 테스트 마스킹 데이터는 NO STACK 임
    test_masking_data_stack_type: StackType = StackType.NO_STACK

    # 데이터 폴더
    train_data_folder_stack_type: StackType = StackType.WITH_SMOTE_NO_STACK # WITH_SMOTE_NO_STACK, STACK, NO_STACK
    test_data_folder_stack_type: StackType = StackType.NO_STACK

    # train path
    # 데이터 세트가 저장된 경로
    train_data_path: str = f"dataset_output_for_train_{train_data_folder_stack_type.value}"

    # 학습 결과가 저장되는 경로
    train_valid_result_path: str = f"outputs/train_result_{train_data_folder_stack_type.value}"

    # test path
    # 데이터 세트가 저장된 경로
    test_data_path: str = f"dataset_output_for_masking_test_{test_data_folder_stack_type.value}"

    # 학습 결과가 저장되는 경로
    test_result_path: str = f"outputs/test_results_{train_data_folder_stack_type.value}" \
        if train_data_folder_stack_type == StackType.WITH_SMOTE_NO_STACK else f"outputs/test_results_{test_data_folder_stack_type.value}"

    test_model_save_path: str = "outputs/train_result_with_smote_no_stack/RAW_RAW_XGBoost_RAW_20250605_185006"


    # 학습 방법 설정
    teacher_data_type: DataType = DataType.RAW
    teacher_frame_type: FrameType = FrameType.RAW_RAW
    teacher_model_type: ModelType = ModelType.XGBOOST
    teacher_model_save_path: str = "outputs/train_result_with_smote_no_stack/RAW_RAW_XGBoost_RAW_20250605_185006"

    student_data_type: DataType = DataType.BERT
    student_frame_type: FrameType = FrameType.EMBEDDING_MIX_LOGITS
    student_model_type: ModelType = ModelType.RESNET_MLP

    # 자동
    is_kd_mode: bool = True if student_frame_type in [FrameType.EMBEDDING_MIX_LOGITS, FrameType.EMBEDDING_LOGITS, FrameType.RAW_LOGITS] else False

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

    # 교사모델 설정
    temperature_t: float = 4.0
    temperature_s: float = 4.0
    alpha: float = 0.5


# CUDA_VISIBLE_DEVICES
# cd /home/juyoung-lab/ws/dev_ws/pi2
# conda activate py310
# python train.py
