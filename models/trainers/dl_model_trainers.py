import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.data.dataloader
import xgboost as xgb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from config.configs import Config, FrameType
from models.trainers.base_trainer import BaseTrainer
from models.models import MLP, ResNetMLP, HybridModel

from utils.input_dim_util import get_input_dim


def load_dl_utils(model_params, cfg: Config):
        ce = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=cfg.scheduler_patience)
        return ce, optim, scheduler

class TorchTrainerBase(BaseTrainer):
    def __init__(
            self,
            cfg: Config,
            input_dim: int = None):
        super().__init__(cfg)
        self.input_dim = input_dim
        self.model = self.build_model().to(cfg.device)
        self.ce, self.optim, self.scheduler = load_dl_utils(self.model.parameters(), self.cfg)
        self.batch_size = cfg.batch_size

    def _update_best(self, val_loss: float, best_valid_loss: float, patience_cnt: int):
        if val_loss < best_valid_loss:
            return val_loss, 0
        else:
            patience_cnt += 1
            return best_valid_loss, patience_cnt


    def _run_epoch(
            self,
            train_loader: DataLoader,
            valid_loader: DataLoader,
        ):

        best_valid_loss = float('inf')
        patience_cnt = 0

        max_epochs = self.cfg.num_epochs
        max_patience = self.cfg.patience

        for epoch in range(max_epochs):
            train_loss = self.run_epoch(train_loader, train=True)
            valid_loss = self.run_epoch(valid_loader, train=False)
            self.scheduler.step(valid_loss)
            best_valid_loss, patience_cnt = self._update_best(valid_loss, best_valid_loss, patience_cnt)
            if patience_cnt >= max_patience:
                print(f"Early stopping at epoch {epoch}, best val loss: {best_valid_loss}")
                break

    def get_loss(self, logits, y):
        loss = self.ce(logits, y)
        return loss

    def make_loader(self, X, y, batch_size: int, shuffle: bool):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def run_epoch(self, data_loader: DataLoader, train: bool = True):
            device = self.cfg.device
            self.model.train() if train else self.model.eval()

            total_loss, n = 0.0, 0

            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                if train: self.optim.zero_grad()
                with torch.set_grad_enabled(train):
                    logits = self.model(X)
                    loss = self.get_loss(logits, y)
                    if train:
                        loss.backward()
                        self.optim.step()

                total_loss += loss.item() * X.size(0)
                n += X.size(0)

            return total_loss / n

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kargs) -> None:
        y_train = y_train.squeeze()
        y_valid = y_valid.squeeze()

        train_loader = self.make_loader(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_loader = self.make_loader(
            X_valid, y_valid,
            batch_size=self.batch_size,
            shuffle=False
        )

        # _run_epoch 메서드 호출 필수
        self._run_epoch(train_loader, valid_loader)


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.cfg.device)
            logits = self.model(X)
        return logits.argmax(dim=1).cpu().numpy()


    def eval(self, X, y, **kwargs):
        return classification_report(y, self.predict(X), **kwargs)

    def save_model(self, path, name):
        torch.save(self.model.state_dict(), os.path.join(path, f"{name}.pth"))

    def load_model(self, path, name):
        self.model.load_state_dict(torch.load(os.path.join(path, f"{name}.pth"), map_location=self.cfg.device, weights_only=True))



class MLPTrainer(TorchTrainerBase):
    def build_model(self):
        return MLP(
            input_dim=self.input_dim,
            hidden_dims=[256] * 2,
            output_dim=4,
            dropout=self.cfg.dropout,
            use_batchnorm=True
        )


class ResNetTrainer(TorchTrainerBase):
    def build_model(self):
        return ResNetMLP(
            input_dim=self.input_dim,
            hidden_dim=256,
            num_blocks=2,
            output_dim=4,
            dropout=self.cfg.dropout,
            use_batchnorm=True
        )


class HybridTrainer(TorchTrainerBase):
    def build_model(self):
        return HybridModel(
            input_dim=self.input_dim,
            output_dim=4,
            dropout=self.cfg.dropout
        )













# class MLPTrainer(BaseTrainer):
#     def __init__(self, config: Config):
#         super().__init__(config)
#         self.build_model()

#     def build_model(self):
#         pass

#     def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kargs) -> None:
#         pass

#     def predict(self, X):
#         pass

#     def eval(self, X, y, **kwargs):
#         pass

#     def save_model(self, path, name):
#         pass

#     def load_model(self, path):
#         pass
