import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.dl_model_trainers import TorchTrainerBase
from models.models import MLP, ResNetMLP, HybridModel
from config.configs import Config
import os
from sklearn.metrics import classification_report
from typing import Union



class TeacherWrapper(nn.Module):
    def __init__(self, teacher, device):
        super().__init__()
        self.teacher, self.device = teacher, device
        self.is_dl = isinstance(teacher, nn.Module)       # DL ↔ ML 구분

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.is_dl:            # DL → 로짓 반환
            return self.teacher(x.to(self.device)), False  # (tensor, is_prob)
        proba = self.teacher.predict_proba(x.cpu().numpy())
        return torch.from_numpy(proba).to(self.device), True  # (tensor, is_prob)


class DistillationTrainer(TorchTrainerBase):
    def __init__(self, cfg: Config, teacher_model):
        super().__init__(cfg)

        # 딥러닝 또는 머신러닝 모델로 교사 모델 선택
        self.teacher = TeacherWrapper(teacher_model, cfg.device).eval()

        for p in self.teacher.parameters():
            if hasattr(p, 'requires_grad'):
                p.requires_grad = False

        self.T_t = cfg.temperature_t
        self.T_s = cfg.temperature_s
        self.alpha = cfg.alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def distill_loss(self, t_out, is_prob, s_logits, y):
        ce_loss = self.ce(s_logits, y)

        # 교사 출력 형태에 따라 분기
        if is_prob:                          # 확률 분포 그대로 사용
            p_t = t_out                      # (B, C)
        else:                                # 로짓이라면 softmax + 온도
            p_t = nn.functional.softmax(t_out / self.T_t, dim=1)

        log_p_s = nn.functional.log_softmax(s_logits / self.T_s, dim=1)
        kd_loss = self.kl(log_p_s, p_t) * (self.T_s ** 2)
        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss

    def __make_loader(self, X_t, X_s, y, shuffle: bool):
        ds = TensorDataset(
            torch.tensor(X_t, dtype=torch.float32),
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y.squeeze(), dtype=torch.long)
        )
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)


    def __run_epoch(self, loader: DataLoader, train: bool = True):
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
                loss, _, _ = self.distill_loss(t_out, is_prob, s_out, y)
                if train:
                    loss.backward()
                    self.optim.step()

            total_loss += loss.item() * X_s.size(0)
            n += X_s.size(0)

        return total_loss / n


    def __update_best(self, val_loss: float, best_val: float, patience_cnt: int):
        if val_loss < best_val:
            return val_loss, 0
        else:
            patience_cnt += 1
            return best_val, patience_cnt


    def fit(self, X_t, X_s, y_train, X_valid_t=None, X_valid_s=None, y_valid=None, **kargs):
        y_train = y_train.squeeze()
        y_valid = y_valid.squeeze()
        train_loader = self.__make_loader(X_t, X_s, y_train, shuffle=True)
        valid_loader = self.__make_loader(X_valid_t, X_valid_s, y_valid, shuffle=False)

        best_val, patience_cnt = float('inf'), 0

        for epoch in range(self.cfg.num_epochs):
            tr_loss = self.__run_epoch(train_loader, train=True)
            val_loss = self.__run_epoch(valid_loader, train=False)
            self.scheduler.step(val_loss)
            best_val, patience_cnt = self.__update_best(val_loss, best_val, patience_cnt)
            if patience_cnt >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch}, best val loss: {best_val:.4f}")
                break


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


# class DistillationTrainer(TorchTrainerBase):
#     def __init__(self, cfg, teacher_model: nn.Module, temperature: float = 3.0, alpha: float = 0.7):
#         super().__init__(cfg)
#         self.teacher = teacher_model.to(cfg.device).eval()
#         for p in self.teacher.parameters(): p.requires_grad = False
#         self.temperature = temperature
#         self.alpha = alpha
#         self.ce = nn.CrossEntropyLoss()
#         self.kl = nn.KLDivLoss(reduction='batchmean')

#     def build_model(self):                       # 학생 모델 정의
#         return HybridModel(                      # 다른 학생을 쓰려면 변경
#             input_dim=self.input_dim,
#             output_dim=4,
#             dropout=self.cfg.dropout
#         )

#     def distill_loss(self, student_logits, teacher_logits, y_true):
#         t = self.temperature
#         kd_loss = self.kl(
#             nn.functional.log_softmax(student_logits / t, dim=1),
#             nn.functional.softmax(teacher_logits / t, dim=1)
#         ) * (t * t)
#         ce_loss = self.ce(student_logits, y_true)
#         return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

#     def fit(self, X_t, X_s, y_s, X_valid_t=None, X_valid_s=None, y_valid=None, **kargs):
#         y_s = y_s.squeeze()
#         train_ds = TensorDataset(
#             torch.tensor(X_t, dtype=torch.float32),
#             torch.tensor(X_s, dtype=torch.float32),
#             torch.tensor(y_s, dtype=torch.long)
#         )
#         if X_valid_t is not None and X_valid_s is not None and y_valid is not None:
#             y_valid = y_valid.squeeze()
#             valid_ds = TensorDataset(
#                 torch.tensor(X_valid_t, dtype=torch.float32),
#                 torch.tensor(X_valid_s, dtype=torch.float32),
#                 torch.tensor(y_valid, dtype=torch.long)
#             )
#         else:
#             raise ValueError("Validation 세트가 필요합니다.")
#         train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
#         valid_loader = DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False)
#         best_val, patience_cnt = float('inf'), 0
#         for epoch in range(self.cfg.num_epochs):
#             self.model.train()
#             for x_t, x_s, y in train_loader:
#                 x_t, x_s, y = x_t.to(self.cfg.device), x_s.to(self.cfg.device), y.to(self.cfg.device)
#                 self.optim.zero_grad()
#                 with torch.no_grad():
#                     teacher_out = self.teacher(x_t)
#                 student_out = self.model(x_s)
#                 loss = self.distill_loss(student_out, teacher_out, y)
#                 loss.backward()
#                 self.optim.step()
#             # ── 검증
#             self.model.eval()
#             tot_loss, n = 0.0, 0
#             with torch.no_grad():
#                 for x_t, x_s, y in valid_loader:
#                     x_t, x_s, y = x_t.to(self.cfg.device), x_s.to(self.cfg.device), y.to(self.cfg.device)
#                     teacher_out = self.teacher(x_t)
#                     student_out = self.model(x_s)
#                     v_loss = self.distill_loss(student_out, teacher_out, y)
#                     tot_loss += v_loss.item() * x_s.size(0)
#                     n += x_s.size(0)
#             val_loss = tot_loss / n
#             self.scheduler.step(val_loss)
#             if val_loss < best_val:
#                 best_val, patience_cnt = val_loss, 0
#             else:
#                 patience_cnt += 1
#             if patience_cnt >= self.cfg.patience:
#                 print(f"Early stopping at epoch {epoch}, best val loss: {best_val:.4f}")
#                 break
