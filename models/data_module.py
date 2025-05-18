import os, numpy as np, pandas as pd, torch
from typing import Tuple, Dict, Any
from config.configs import DataType, FrameType, StackType
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

