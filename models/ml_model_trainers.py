import xgboost as xgb
from models.base_trainer import BaseTrainer
from config.configs import Config
from sklearn.metrics import classification_report
import os
import lightgbm as lgb
import joblib

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



class LightGBMTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model = self.build_model()

    def build_model(self):
        return lgb.LGBMClassifier(
            n_estimators=self.cfg.num_epochs,
            learning_rate=self.cfg.lr,
            num_classes=4,
            max_depth=self.cfg.max_depth,
            objective='multiclass',
            random_state=self.cfg.seed,
            subsample=self.cfg.subsample,
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs) -> None:
        # verbose = kwargs.get("verbose", False)
        # verbose = 0 if verbose == False else 10
        if y_train.ndim > 1:
            y_train = y_train.ravel()
        if y_valid is not None and y_valid.ndim > 1:
            y_valid = y_valid.ravel()

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='multi_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.cfg.early_stopping_rounds),
            ],
        )

    def predict(self, X):
        return self.model.predict(X)

    def eval(self, X, y, **kwargs):
        if y.ndim > 1:
            y = y.ravel()
        return classification_report(y, self.predict(X), **kwargs)

    def save_model(self, path, name):
        joblib.dump(self.model, os.path.join(path, f"{name}.pkl"))

    def load_model(self, path, name):
        self.model = joblib.load(os.path.join(path, f"{name}.pkl"))
