from config.configs import Config, ModelType
from typing import Any, Dict, List, Tuple, Type
from models.trainers.ml_model_trainers import XGBoostTrainer, LightGBMTrainer
from models.trainers.base_trainer import BaseTrainer
from models.trainers.dl_model_trainers import MLPTrainer, ResNetTrainer, TorchTrainerBase
from models.trainers.tabnet_trainer import TabNetTrainer
from models.trainers.dist_trainers import DistillResNetTrainer, DistillMLPTrainer, DistillationTrainer

from utils.input_dim_util import get_input_dim

class TrainerFactory:
    _registry: Dict[ModelType, Type[BaseTrainer]] = {
        ModelType.XGBOOST: XGBoostTrainer,
        ModelType.LIGHTGBM: LightGBMTrainer,
        ModelType.TABNET: TabNetTrainer,
        ModelType.MLP: MLPTrainer,
        ModelType.RESNET_MLP: ResNetTrainer,
    }

    _registry_distill: Dict[ModelType, Type[BaseTrainer]] = {
        ModelType.RESNET_MLP: DistillResNetTrainer,
        ModelType.MLP: DistillMLPTrainer,
    }

    @classmethod
    def build(self, model_type: ModelType, cfg: Config) -> BaseTrainer:
        trainer_cls = self._registry[model_type]

        if issubclass(trainer_cls, TorchTrainerBase):
            input_dim = get_input_dim(cfg.student_frame_type, cfg)
            return trainer_cls(cfg, input_dim)
        else:
            return trainer_cls(cfg)

    @classmethod
    def build_distill_trainer(
        self,
        student_type: ModelType,
        teacher_type: ModelType,
        cfg: Config,
        scen: int = 13
    ) -> DistillationTrainer:

        # --- 1. 교사(Teacher) 모델 인스턴스 생성 ---
        teacher_trainer_cls = self._registry[teacher_type]
        teacher_trainer: BaseTrainer

        if issubclass(teacher_trainer_cls, TorchTrainerBase):
            # 교사가 DL 모델일 경우, '교사의 FrameType'으로 input_dim 계산 후 주입
            teacher_input_dim = get_input_dim(cfg.teacher_frame_type, cfg)
            teacher_trainer = teacher_trainer_cls(cfg, teacher_input_dim)
        else:
            # 교사가 ML 모델(XGBoost 등)일 경우, 그대로 생성
            teacher_trainer = teacher_trainer_cls(cfg)


        # --- 2. 학생(Student) 모델 인스턴스 생성 ---
        student_trainer_cls = self._registry_distill[student_type]

        # '학생의 FrameType'으로 input_dim 계산
        student_input_dim = get_input_dim(cfg.student_frame_type, cfg)

        # --- 3. 최종 DistillationTrainer 생성 및 반환 ---
        # 생성자에 준비된 '교사 인스턴스'와 '학생의 input_dim'을 전달
        return student_trainer_cls(
            cfg=cfg,
            teacher_trainer=teacher_trainer,  # 클래스가 아닌 '인스턴스' 전달
            scen=scen,
            student_input_dim=student_input_dim
        )
