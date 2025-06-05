import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.trainers.dl_model_trainers import TorchTrainerBase
from models.models import MLP, ResNetMLP, HybridModel
from config.configs import Config
import os
from sklearn.metrics import classification_report
from typing import Union, Any
from models.trainers.teacher_model_wrapper import TeacherWrapper




class DistillationTrainer(TorchTrainerBase):
    def __init__(
            self,
            cfg: Config,
            teacher_trainer: TorchTrainerBase,
            scen: int = 13,
            student_input_dim: int = None
        ):
        super().__init__(cfg, input_dim=student_input_dim)

        # 딥러닝 또는 머신러닝 모델로 교사 모델 선택

        teacher_trainer.load_model(os.path.join(cfg.teacher_model_save_path, "model_save"), f"scen_{scen}")
        self.teacher = TeacherWrapper(teacher_trainer, cfg.device).eval()

        for p in self.teacher.parameters():
            if hasattr(p, 'requires_grad'):
                p.requires_grad = False

        self.T_t = cfg.temperature_t
        self.T_s = cfg.temperature_s
        self.alpha = cfg.alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def get_loss(self, t_out, is_prob, s_logits, y):
        ce_loss = self.ce(s_logits, y)

        # 교사 출력 형태에 따라 분기
        if is_prob:                          # 확률 분포 그대로 사용
            p_t = t_out                      # (B, C)
        else:                                # 로짓이라면 softmax + 온도
            p_t = nn.functional.softmax(t_out / self.T_t, dim=1)

        log_p_s = nn.functional.log_softmax(s_logits / self.T_s, dim=1)
        kd_loss = self.kl(log_p_s, p_t) * (self.T_s ** 2)
        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss

    def make_loader(self, X_t, X_s, y, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X_t, dtype=torch.float32),
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y.squeeze(), dtype=torch.long)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


    def run_epoch(self, loader: DataLoader, train: bool = True):
        # 모델 학습 모드 설정
        device = self.cfg.device
        self.model.train() if train else self.model.eval()

        total_loss, n = 0.0, 0

        for X_t, X_s, y in loader:
            X_t, X_s, y = X_t.to(device), X_s.to(device), y.to(device)
            if train: self.optim.zero_grad()
            with torch.set_grad_enabled(train):
                t_out, is_prob = self.teacher(X_t)
                s_out = self.model(X_s)
                loss = self.get_loss(t_out, is_prob, s_out, y)
                if train:
                    loss.backward()
                    self.optim.step()

            total_loss += loss.item() * X_s.size(0)
            n += X_s.size(0)

        return total_loss / n


    def fit(self, X_t, X_s, y_train, X_valid_t=None, X_valid_s=None, y_valid=None, **kargs):
        y_train = y_train.squeeze()
        y_valid = y_valid.squeeze()

        train_loader= self.make_loader(
            X_t, X_s, y_train,
            self.batch_size,
            shuffle=True
        )
        valid_loader = self.make_loader(
            X_valid_t, X_valid_s, y_valid,
            self.batch_size,
            shuffle=False
        )

        # _run_epoch 메서드 호출 필수
        self._run_epoch(train_loader, valid_loader)


    def predict(self, X_s):
        self.model.eval()
        with torch.no_grad():
            X_s = torch.tensor(X_s, dtype=torch.float32).to(self.cfg.device)
            logits = self.model(X_s)
        return logits.argmax(dim=1).cpu().numpy()

    def eval(self, X_s, y, **kwargs):
        return classification_report(y, self.predict(X_s), **kwargs)

    def save_model(self, path, name):
        torch.save(self.model.state_dict(), os.path.join(path, f"{name}.pth"))

    def load_model(self, path, name):
        self.model.load_state_dict(torch.load(os.path.join(path, f"{name}.pth"), map_location=self.cfg.device, weights_only=True))


class DistillResNetTrainer(DistillationTrainer):
    def build_model(self):
        return ResNetMLP(
            input_dim=self.input_dim,
            hidden_dim=256,
            num_blocks=2,
            output_dim=4,
            dropout=self.cfg.dropout,
            use_batchnorm=True
        )

class DistillMLPTrainer(DistillationTrainer):
    def build_model(self):
        return MLP(
            input_dim=self.input_dim,
            hidden_dims=[256] * 2,
            output_dim=4,
            dropout=self.cfg.dropout,
            use_batchnorm=True
        )


