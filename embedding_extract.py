
"""
# 언어 모델의 임베딩을 추출하는 코드

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
    seed: int = 42
    batch_size: Dict[str, int] = field(default_factory=lambda: {"basic": 4, "smote": 2})
    device: str = "cuda"
    models: List[ModelType] = field(default_factory=lambda: [ModelType.GEMMA3])
    dataset_select: List[str] = field(default_factory=lambda: ["FD001", "FD003"])
    dataset_path: str = "engine_knee_plots_multi/all_engines_labeled.csv"
    type: PhaseType = PhaseType.TRAIN
    data_stack_type: DataVersion = DataVersion.NO_STACK
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
    output_dir: str = f"dataset_output_for_{type.value}_{data_stack_type.value}"

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

    df = pd.read_csv(config.dataset_path)
    train_df, test_df = split_dataset(df, config.dataset_select, config.seed)

    X_train, y_train = train_df.drop(columns=config.drop_cols), train_df["state"]
    X_test, y_test = test_df.drop(columns=config.drop_cols), test_df["state"]

    float_cols = X_train.select_dtypes(include="float").columns
    X_train[float_cols] = X_train[float_cols].round(5)
    X_test[float_cols]  = X_test[float_cols].round(5)

    X_train.rename(columns=config.cols_rename_map, inplace=True)
    X_test.rename(columns=config.cols_rename_map, inplace=True)


    # 학습용으로 임베딩 추출 -> 시나리오 0부터 센서 개수(16-1)까지 랜덤으로 결측 마스킹
    if config.type == PhaseType.TRAIN:
        """
        # 계획
        1. LM으로 임베딩을 추출할건데, SMOTE 증강 유무에 따라서 동시에 저장
        2. SMOTE 증강은 5배수 고정

        # 마스킹 규칙
        1. 시나리오 0 ~ 센서 개수-1까지 진행
        2. 각 시나리오의 숫자 의미는 결측되는 센서의 개수를 의미하며 결측되는 센서는 랜덤 (시나리오 0은 결측X)
        3. key(컬럼명): value(값)의 형태인 json string으로 각 행을 저장
        4. LM에 json string을 넣고 임베딩 추출

        # 저장 규칙
        1. 각 LM 이름별로 폴더를 만들고 그 안에 임베딩된 numpy 형태로 저장
        2. raw 폴더 이름으로 원본 숫자 형태로 저장(마스킹 적용한 것)
        3. y에 해당하는 내용은 전부 동일하므로 상위 폴더에 y_train.csv로 저장
        4. SMOTE 증강의 유무에 따라 파일명 뒤에 _smote를 붙임

        """

        train_perm_idx = generate_random_permutation(config.seed, X_train)
        apply_masking_scenario("train", X_train, train_perm_idx, y_train, config)

        test_perm_idx = generate_random_permutation(config.seed, X_test)
        apply_masking_scenario("valid", X_test, test_perm_idx, y_test, config)


    # 테스트용으로 임베딩 추출 -> lgbm feature importance로 시나리오 순서대로 결측 마스킹
    # 테스트용은 valid 데이터만 추출
    else:
        test_perm_idx = generate_feature_importance_permutation(X_test, config)
        apply_masking_scenario("valid", X_test, test_perm_idx, y_test, config)

    printi(f"End of embedding extraction process -> {config.output_dir}")


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
