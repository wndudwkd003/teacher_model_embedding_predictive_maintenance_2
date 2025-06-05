from models.trainers.base_trainer import BaseTrainer
from config.configs import Config
from sklearn.metrics import classification_report
import os
from pytorch_tabnet.tab_model import TabNetClassifier


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
