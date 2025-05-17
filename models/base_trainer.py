from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import numpy as np

from config.configs import Config


class BaseTrainer(ABC):
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.mode: Any = None

    @abstractmethod
    def build_model(self) -> Any:
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray|None, y_valid: np.ndarray|None, **kargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def eval(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def save_model(self, dir: str, name: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str, name) -> None:
        pass

