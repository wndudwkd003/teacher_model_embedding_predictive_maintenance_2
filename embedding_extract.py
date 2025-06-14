
"""
# 언어 모델의 임베딩을 추출하는 코드
# CUDA_VISIBLE_DEVICES
"""

import json
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5Tokenizer, T5EncoderModel
from imblearn.over_sampling import SMOTE

class JsonStringDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class ModelType(Enum):
    BERT = "bert-base-uncased"
    GPT2 = "gpt2"
    GEMMA3 = "google/gemma-3-1b-pt"
    T5 = "t5-base"
    ELECTRA = "google/electra-base-discriminator"


class DataVersion(Enum):
    STACK = "stack"
    NO_STACK = "no_stack"


class PhaseType(Enum):
    TRAIN = "train"
    MASKING_TEST = "masking_test"


@dataclass
class Config:
    type: PhaseType = PhaseType.TRAIN
    models: List[ModelType] = field(default_factory=lambda: [ModelType.BERT])
    apply_smote: bool = True
    data_stack_type: DataVersion = DataVersion.NO_STACK
    smote_multiplier: float = 5.0 # SMOTE로 증강할 비율 (예: 5.0 -> 소수 클래스의 샘플 수를 5배로 증강)
    k_neighbors_smote: int = 5 # SMOTE에서 사용할 k_neighbors 값
    seed: int = 42
    batch_size: Dict[str, int] = field(default_factory=lambda: {"basic": 4, "smote": 4})
    device: str = "cuda"
    dataset_select: List[str] = field(default_factory=lambda: ["FD001", "FD003"])
    dataset_path: str = "engine_knee_plots_multi/all_engines_labeled.csv"
    feature_importance_path: str = "outputs/feature_importance/nasa_dataset/feature_importance.csv"
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

    output_dir_base_name: str = "dataset_output"
    output_dir: str = field(init=False)

    def __post_init__(self):
        self.output_dir = f"dataset_output_for_{self.type.value}_{self.data_stack_type.value}"

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


def main(config: Config):
    # 언어 모델 임베딩 추출

    device = config.device if torch.cuda.is_available() else "cpu"
    printi(f"Using device: {device}")

    if not (config.type == PhaseType.TRAIN and config.apply_smote):
        os.makedirs(config.output_dir, exist_ok=True)
    printi(f"Initial base output directory (before potential SMOTE modification): {config.output_dir}")


    df = pd.read_csv(config.dataset_path)
    train_df, test_df = split_dataset(df, config.dataset_select, config.seed)

    X_train_orig, y_train_orig = train_df.drop(columns=config.drop_cols), train_df["state"]
    X_test_orig, y_test_orig = test_df.drop(columns=config.drop_cols), test_df["state"]

    common_float_cols_train = X_train_orig.select_dtypes(include="float").columns
    X_train_orig[common_float_cols_train] = X_train_orig[common_float_cols_train].round(5)

    common_float_cols_test = X_test_orig.select_dtypes(include="float").columns
    X_test_orig[common_float_cols_test] = X_test_orig[common_float_cols_test].round(5)

    X_train_to_process = X_train_orig.copy()
    y_train_to_process = y_train_orig.copy() # Series 형태
    X_test_to_process = X_test_orig.copy()
    y_test_to_process = y_test_orig.copy() # Series 형태

    original_config_output_dir = config.output_dir # 원래 output_dir 저장

    if config.type == PhaseType.TRAIN:
        train_phase_name = "train"
        y_df_for_train_scenario = y_train_to_process.to_frame(name='state') # apply_masking_scenario의 시그니처에 맞춤

        if config.apply_smote:
            printi("Applying SMOTE to training data...")

            # SMOTE용 output_dir 설정 (기존 config.output_dir 임시 변경)
            smote_phase_descriptor = config.type.value + "_with_smote" # "train_with_smote"
            config.output_dir = f"{config.output_dir_base_name}_for_{smote_phase_descriptor}_{config.data_stack_type.value}"
            os.makedirs(config.output_dir, exist_ok=True)
            printi(f"Output directory for SMOTE data temporarily set to: {config.output_dir}")

            value_counts = y_train_to_process.value_counts()
            min_class_label = value_counts.idxmin()
            min_class_count = value_counts.min()

            k_neighbors = min(config.k_neighbors_smote, min_class_count - 1) if min_class_count > 1 else 1

            target_minority_samples = int(min_class_count * config.smote_multiplier)

            sampling_strategy_dict = {min_class_label: target_minority_samples}
            smote = SMOTE(random_state=config.seed, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy_dict)

            printi(f"SMOTE params: k_neighbors={k_neighbors}, sampling_strategy={sampling_strategy_dict}")
            X_train_smote_np, y_train_smote_series = smote.fit_resample(X_train_to_process, y_train_to_process)

            X_train_to_process = pd.DataFrame(X_train_smote_np, columns=X_train_to_process.columns)
            y_train_to_process = pd.Series(y_train_smote_series, name=y_train_to_process.name) # Series로 유지
            y_df_for_train_scenario = y_train_to_process.to_frame(name='state') # 변경된 y로 DataFrame 업데이트
            printi(f"Training data shape after SMOTE: X={X_train_to_process.shape}, y={y_train_to_process.shape}")

        X_train_to_process.rename(columns=config.cols_rename_map, inplace=True)
        train_perm_idx = generate_random_permutation(config.seed, X_train_to_process)
        apply_masking_scenario(train_phase_name, X_train_to_process, train_perm_idx, y_df_for_train_scenario, config)

        # Validation 데이터 처리
        X_test_to_process.rename(columns=config.cols_rename_map, inplace=True)
        valid_perm_idx = generate_random_permutation(config.seed, X_test_to_process)
        apply_masking_scenario("valid", X_test_to_process, valid_perm_idx, y_test_to_process.to_frame(name='state'), config)


    else: # PhaseType.MASKING_TEST
        # MASKING_TEST 시에는 SMOTE 적용 안 함, 원래 config.output_dir 사용
        config.output_dir = original_config_output_dir # 혹시 모르니 복원 (실제로는 TRAIN 블록을 안 타므로 변경 안됐을 것)
        X_test_to_process.rename(columns=config.cols_rename_map, inplace=True)
        test_perm_idx = generate_feature_importance_permutation(X_test_to_process, config)
        apply_masking_scenario("valid", X_test_to_process, test_perm_idx, y_test_to_process.to_frame(name='state'), config)

    printi(f"End of embedding extraction process. Final base output directory was: {config.output_dir}")

# np.concatenate([raw_masked_results[scen-1], masked_X], axis=0) if scen > 0 else masked_X

def apply_masking_scenario(phase: str, X_df: pd.DataFrame, train_perm_idx: np.ndarray, y_df: pd.DataFrame, config: Config):
    row_count = X_df.shape[0]

    def get_scen_raw_path(scen:int, phase:str):
            return os.path.join(config.output_dir, phase, f"iteration_{scen}")

    for scen in range(len(X_df.columns)):
        printi(f"Current Scenario: {scen}")
        masked_X = X_df.to_numpy().copy()
        masked_idx = train_perm_idx[:, :scen]
        masked_X[np.arange(row_count)[:, None], masked_idx] = -1
        masked_raw_X_df = pd.DataFrame(masked_X, columns=X_df.columns)
        current_scen_X_df = masked_raw_X_df.copy()

        # raw 데이터 저장
        raw_dir = get_scen_raw_path(scen, "RAW")
        os.makedirs(raw_dir, exist_ok=True)

        if config.data_stack_type == DataVersion.STACK:
            if scen > 0:
                before_dir = get_scen_raw_path(scen-1, "RAW")
                before_masked_X = pd.read_csv(os.path.join(before_dir, f"X_{phase}.csv"))
                masked_raw_X_df = pd.DataFrame(pd.concat([before_masked_X, masked_raw_X_df], axis=0), columns=X_df.columns)

            y_df = pd.DataFrame(pd.concat([y_df]*(scen+1), axis=0), columns=["state"])

        masked_raw_X_df.to_csv(os.path.join(raw_dir, f"X_{phase}.csv"), index=False)
        y_df.to_csv(os.path.join(raw_dir, f"y_{phase}.csv"), index=False)

        # json 데이터 변환
        json_strings = [json.dumps(row) for row in current_scen_X_df.to_dict(orient="records")]
        printi(f"length: {len(json_strings)}")

        # 임베딩 추출
        for model in config.models:
            raw_dir = get_scen_raw_path(scen, model.name)
            embeddings = extract_lm_embedding(json_strings, model, config.device)

            if config.data_stack_type == DataVersion.STACK:
                if scen > 0:
                    before_dir = get_scen_raw_path(scen-1, model.name)
                    before_masked_X = np.load(os.path.join(before_dir, f"X_{phase}.npy"))
                    embeddings = np.concatenate([before_masked_X, embeddings], axis=0)

            embedding_dir = os.path.join(config.output_dir, model.name, f"iteration_{scen}")
            os.makedirs(embedding_dir, exist_ok=True)
            np.save(os.path.join(embedding_dir, f"X_{phase}.npy"), embeddings)



def extract_lm_embedding(json_strings: List[str], model_type: ModelType, device: str) -> np.ndarray:
    # Load the tokenizer and model
    if model_type in [ModelType.BERT, ModelType.ELECTRA]:
        tokenizer = AutoTokenizer.from_pretrained(model_type.value)
        model = AutoModel.from_pretrained(model_type.value)
    elif model_type in [ModelType.GPT2, ModelType.GEMMA3]:
        tokenizer = AutoTokenizer.from_pretrained(model_type.value)
        model = AutoModelForCausalLM.from_pretrained(model_type.value)
        if model_type == ModelType.GPT2:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
    elif model_type == ModelType.T5:
        tokenizer = T5Tokenizer.from_pretrained(model_type.value)
        model = T5EncoderModel.from_pretrained(model_type.value)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    def collate(batch):
        return tokenizer(batch, return_tensors="pt", padding="longest", truncation=True, max_length=300)

    loader = torch.utils.data.DataLoader(
        JsonStringDataset(json_strings),
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate
    )

    """
    # 임베딩 추출
    # 접근 방법은 관련 논문을 따름

    """
    embeddings = []

    with torch.no_grad():
        model.to(device).eval()

        for inputs in tqdm(loader, desc=f"Extracting {model_type.name}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            last = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)          # (batch,L,1)
            summed   = (last * mask).sum(dim=1)                    # PAD 제외 합
            lengths  = mask.sum(dim=1).clamp(min=1)                # 0 나누기 방지
            batch_emb = (summed / lengths).cpu().numpy()           # (batch, hidden)

            embeddings.append(batch_emb)

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings   # np.ndarray


def generate_random_permutation(seed: int, X: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    rand_mat = rng.random(tuple(X.shape))
    perm = np.argsort(rand_mat, axis=1)
    return perm


def generate_feature_importance_permutation(X: pd.DataFrame, config: Config) -> np.ndarray:
    importance = pd.read_csv(config.feature_importance_path)
    idx = X.columns.get_indexer(importance["feature"])
    perm = np.tile(idx, (len(X), 1))
    return perm


if __name__ == "__main__":
    config = Config()
    seed_everything(config.seed)
    main(config)
