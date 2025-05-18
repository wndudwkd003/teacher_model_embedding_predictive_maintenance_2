from config.configs import Config, ModelType
from typing import Any, Dict, List, Tuple, Type
from models.ml_model_trainers import XGBoostTrainer, LightGBMTrainer
from models.base_trainer import BaseTrainer
from models.dl_model_trainers import MLPTrainer, ResNetTrainer, TabNetTrainer



class TrainerFactory:
    _registry: Dict[ModelType, Type[BaseTrainer]] = {
        ModelType.XGBOOST: XGBoostTrainer,
        ModelType.LIGHTGBM: LightGBMTrainer,
        ModelType.TABNET: TabNetTrainer,
        ModelType.MLP: MLPTrainer,
        ModelType.RESNET_MLP: ResNetTrainer,
    }

    @classmethod
    def build(self, model_type: ModelType, cfg: Config) -> BaseTrainer:
        trainer_cls = self._registry[model_type]
        return trainer_cls(cfg)
