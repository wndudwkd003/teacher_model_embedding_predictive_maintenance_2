from typing import Any
import torch
from torch import nn


class TeacherWrapper(nn.Module):
    def __init__(self, trainer: Any, device):
        super().__init__()
        self.device = device

        # Trainer 인스턴스일 경우: .model 속성을 사용
        if hasattr(trainer, 'model'):
            self.teacher = trainer.model
        else:
            self.teacher = trainer

        # DL 모델인지 ML 모델인지 분기
        self.is_dl = isinstance(self.teacher, nn.Module)

        if self.is_dl:
            self.teacher.to(self.device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.is_dl:
            return self.teacher(x.to(self.device)), False
        else:
            proba = self.teacher.predict_proba(x.cpu().numpy())
            return torch.from_numpy(proba).to(self.device), True
