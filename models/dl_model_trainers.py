import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.data.dataloader
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report

from config.configs import Config, FrameType
from models.base_trainer import BaseTrainer
from models.models import MLP, ResNetMLP


class TabNetTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model = self.build_model()

    def build_model(self):
        return TabNetClassifier(
            n_d         = 32,
            n_a         = 32,
            n_steps     = 5,
            gamma       = 1.3,
            seed        = self.cfg.seed,
            device_name = self.cfg.device,
            verbose     = 0,
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kargs) -> None:
        if y_train.ndim > 1:
            y_train = y_train.ravel()
        if y_valid is not None and y_valid.ndim > 1:
            y_valid = y_valid.ravel()

        self.model.fit(
            X_train, y_train,
            eval_set    = [(X_valid, y_valid)],
            eval_metric = ["accuracy"],
            max_epochs  = self.cfg.num_epochs,
            patience    = self.cfg.patience,
            batch_size  = self.cfg.batch_size,
        )

    def predict(self, X):
        return self.model.predict(X)

    def eval(self, X, y, **kwargs):
        if y.ndim > 1:
            y = y.ravel()
        return classification_report(y, self.predict(X), **kwargs)

    def save_model(self, path, name):
        self.model.save_model(os.path.join(path, name))

    def load_model(self, path, name=None):
        self.model.load_model(os.path.join(path, f"{name}.zip"))


def load_dl_utils(model_params, cfg: Config):
        loss_fun = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=cfg.scheduler_patience)
        return loss_fun, optim, scheduler



def get_input_dim(cfg: Config):
    if cfg.student_frame_type in [FrameType.RAW_RAW, FrameType.RAW_LOGITS]:
        return cfg.raw_input_dim
    elif cfg.student_frame_type in [FrameType.EMBEDDING_RAW, FrameType.EMBEDDING_LOGITS]:
        return cfg.embedding_input_dim
    elif cfg.student_frame_type == FrameType.EMBEDDING_MIX_RAW:
        return cfg.mix_input_dim
    else:
        raise ValueError(f"Unknown input dimension type: {cfg.student_frame_type}. Please check the configuration.")


class TorchTrainerBase(BaseTrainer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.input_dim = get_input_dim(cfg)
        self.model = self.build_model().to(cfg.device)
        self.loss_fun, self.optim, self.scheduler = load_dl_utils(self.model.parameters(), self.cfg)

    def __run_epoch(self, data_loader, train=True):
            self.model.train() if train else self.model.eval()
            total_loss, n = 0.0, 0
            for X, y in data_loader:
                X, y = X.to(self.cfg.device), y.to(self.cfg.device)
                if train:
                    self.optim.zero_grad()
                with torch.set_grad_enabled(train):
                    out = self.model(X)
                    loss = self.loss_fun(out, y)
                    if train:
                        loss.backward()
                        self.optim.step()
                total_loss += loss.item() * X.size(0)
                n += X.size(0)
            return total_loss / n

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kargs) -> None:
        y_train = y_train.squeeze()
        y_valid = y_valid.squeeze()
        train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        if X_valid is not None and y_valid is not None:
            valid_ds = torch.utils.data.TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))
        else:
            raise ValueError("Validation data is required for MLPTrainer.")
        best_val = float('inf')
        patience_cnt = 0
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False)
        for epoch in range(self.cfg.num_epochs):
            train_loss = self.__run_epoch(train_loader, train=True)
            valid_loss = self.__run_epoch(valid_loader, train=False)
            self.scheduler.step(valid_loss)
            if valid_loss < best_val:
                best_val = valid_loss
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch}, best val loss: {best_val}")
                break


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.cfg.device)
            out = self.model(X)
        return out.argmax(dim=1).cpu().numpy()


    def eval(self, X, y, **kwargs):
        return classification_report(y, self.predict(X), **kwargs)

    def save_model(self, path, name):
        torch.save(self.model.state_dict(), os.path.join(path, f"{name}.pth"))

    def load_model(self, path, name):
        self.model.load_state_dict(torch.load(os.path.join(path, f"{name}.pth"), map_location=self.cfg.device, weights_only=True))





class MLPTrainer(TorchTrainerBase):
    def build_model(self):
        return MLP(input_dim=self.input_dim, hidden_dims=[256] * 2, output_dim=4, dropout=self.cfg.dropout, use_batchnorm=True)


class ResNetTrainer(TorchTrainerBase):
    def build_model(self):
        return ResNetMLP(input_dim=self.input_dim, hidden_dim=256, num_blocks=2, output_dim=4, dropout=self.cfg.dropout, use_batchnorm=True)












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
