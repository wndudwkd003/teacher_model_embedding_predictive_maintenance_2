import xgboost as xgb
from models.base_trainer import BaseTrainer
from config.configs import Config
from sklearn.metrics import classification_report
import os

class XGBoostTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model = self.build_model()

    def build_model(self):
        return xgb.XGBClassifier(
            device=self.cfg.xgb_device,
            tree_method=self.cfg.tree_method,
            random_state=self.cfg.seed,
            eval_metric=self.cfg.eval_metric,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            n_estimators=self.cfg.num_epochs,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.lr,
            subsample=self.cfg.subsample,
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs) -> None:
        verbose = kwargs.get("verbose", False)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)] if X_valid is not None else None,
            verbose=verbose,
        )

    def predict(self, X):
        return self.model.predict(X)

    def eval(self, X, y, **kwargs):
        return classification_report(y, self.predict(X), **kwargs)

    def save_model(self, path, name):
        self.model.save_model(os.path.join(path, f"{name}.json"))

    def load_model(self, path, name):
        self.model.load_model(os.path.join(path, f"{name}.json"))
