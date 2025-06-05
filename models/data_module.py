import os, numpy as np, pandas as pd, torch
from typing import Tuple, Dict, Any
from config.configs import DataType, FrameType, StackType
from config.configs import Config
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load(self, dir: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
        pass

class CSVLoader(BaseLoader):
    def load(self, dir, name):
        X = pd.read_csv(os.path.join(dir, f"X_{name}.csv")).values
        y = pd.read_csv(os.path.join(dir, f"y_{name}.csv")).values
        return X, y

class NPYLoader(BaseLoader):
    def load(self, dir, name):
        X = np.load(os.path.join(dir, f"X_{name}.npy"))
        return X, None


class DataFactory:
    _registry: Dict[DataType, BaseLoader] = {
        DataType.RAW: CSVLoader,   # ← 클래스 그대로
        DataType.BERT: NPYLoader,
        DataType.GPT2: NPYLoader,
        DataType.GEMMA3: NPYLoader,
        DataType.ELECTRA: NPYLoader,
        DataType.T5: NPYLoader,
    }

    @classmethod
    def load(cls, root: str, name: str, scen: int, dtype: DataType, stack_type: StackType = StackType.STACK):
        X = []
        y = []
        loader_cls = cls._registry[dtype]
        loader = loader_cls()

        if stack_type == StackType.STACK:
            for i in range(scen+1):
                folder = os.path.join(root, dtype.name, f"iteration_{i}")
                X_i, y_i = loader.load(folder, name)
                X.append(X_i)
                y.append(y_i)
        else:
            folder = os.path.join(root, dtype.name, f"iteration_{scen}")
            X_i, y_i = loader.load(folder, name)
            X.append(X_i)
            y.append(y_i)

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0) if y[0] is not None else None

        return X, y



class DataManager:
    @staticmethod
    def __prepare_features(
        config: Config,
        scen: int,
        raw_data: np.ndarray,
        phase: str = "train",
        stack_type: StackType = StackType.STACK,
        frame_type: FrameType = FrameType.RAW_RAW,
        data_type: DataType = DataType.RAW,
    ):
        # FrameType에 따라 최종 입력 데이터를 구성
        if frame_type == FrameType.RAW_RAW:
            return raw_data

        # 임베딩 데이터 로드
        embedding_data, _ = DataFactory.load(
            config.train_data_path,
            phase,
            scen,
            data_type,
            stack_type
        )

        if frame_type in [FrameType.EMBEDDING_RAW, FrameType.EMBEDDING_LOGITS]:
            return embedding_data

        elif frame_type in [FrameType.EMBEDDING_MIX_RAW, FrameType.EMBEDDING_MIX_LOGITS]:
            return np.concatenate([raw_data, embedding_data], axis=1)


    @staticmethod
    def get_train_data(
        current_pipe_path: str,
        config: Config,
        scen: int
    ):
        # 현재 파이프라인 경로에 따라 데이터 스택 타입을 결정
        train_data_stack_type = config.train_data_stack_type
        test_data_stack_type = config.test_data_stack_type

        X_train_raw, y_train = DataFactory.load(current_pipe_path, "train", scen, DataType.RAW, train_data_stack_type)
        X_valid_raw, y_valid = DataFactory.load(current_pipe_path, "valid", scen, DataType.RAW, test_data_stack_type)

        if not config.is_kd_mode:
            X_train = DataManager.__prepare_features(
                config, scen, X_train_raw, "train", train_data_stack_type, config.student_frame_type, config.student_data_type
            )
            X_valid = DataManager.__prepare_features(
                config, scen, X_valid_raw, "valid", test_data_stack_type, config.student_frame_type, config.student_data_type
            )
            return (X_train, y_train), (X_valid, y_valid)

        else:
            X_train_teacher = DataManager.__prepare_features(
                config, scen, X_train_raw, "train", train_data_stack_type, config.teacher_frame_type, config.teacher_data_type
            )
            X_valid_teacher = DataManager.__prepare_features(
                config, scen, X_valid_raw, "valid", test_data_stack_type, config.teacher_frame_type, config.teacher_data_type
            )

            X_train_student = DataManager.__prepare_features(
                config, scen, X_train_raw, "train", train_data_stack_type, config.student_frame_type, config.student_data_type
            )
            X_valid_student = DataManager.__prepare_features(
                config, scen, X_valid_raw, "valid", test_data_stack_type, config.student_frame_type, config.student_data_type
            )
            return (X_train_teacher, X_train_student, y_train), (X_valid_teacher, X_valid_student, y_valid)


    @staticmethod
    def get_masking_test_data(
        current_pipe_path: str,
        config: Config,
        scen: int
    ):
        test_data_stack_type = config.test_masking_data_stack_type

        X_valid_raw, y_valid = DataFactory.load(current_pipe_path, "valid", scen, DataType.RAW, test_data_stack_type)

        if not config.is_kd_mode:

            X_valid = DataManager.__prepare_features(
                config, scen, X_valid_raw, "valid", test_data_stack_type, config.student_frame_type, config.student_data_type
            )
            return X_valid, y_valid

        else:
            X_valid_teacher = DataManager.__prepare_features(
                config, scen, X_valid_raw, "valid", test_data_stack_type, config.teacher_frame_type, config.teacher_data_type
            )

            X_valid_student = DataManager.__prepare_features(
                config, scen, X_valid_raw, "valid", test_data_stack_type, config.student_frame_type, config.student_data_type
            )
            return X_valid_teacher, X_valid_student, y_valid

