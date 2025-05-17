"""
# xgboost Feature Importance 추출해서 그래프로 시각화
# csv 파일로 저장

"""

import xgboost as xgb
import pandas as pd
import os
from config.configs import Config
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

DATA_DIR = "dataset_output_for_train/RAW/iteration_0"
OUT_DIR = "outputs"


def load_csv(name: str):
    return pd.read_csv(os.path.join(DATA_DIR, name))

def main():
    cfg = Config()

    X_train = load_csv("X_train.csv")
    y_train = load_csv("y_train.csv")
    X_valid = load_csv("X_valid.csv")
    y_valid = load_csv("y_valid.csv")

    model = xgb.XGBClassifier(
        device=cfg.xgb_device,
        tree_method=cfg.tree_method,
        random_state=cfg.seed,
        eval_metric=cfg.eval_metric,
        early_stopping_rounds=cfg.early_stopping_rounds,
        n_estimators=cfg.num_epochs,
        max_depth=cfg.max_depth,
        learning_rate=cfg.lr,
        subsample=cfg.subsample,
    )

    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_valid.values, y_valid.values)],
        verbose=True,
    )

    print(classification_report(y_valid.values, model.predict(X_valid.values), digits=5))

    # directory
    description = "nasa_dataset"
    output_dir = os.path.join(OUT_DIR, "feature_importance", f"{description}")
    os.makedirs(output_dir, exist_ok=True)

    # Feature importance
    importance = model.feature_importances_

    columns = X_train.columns
    feature_importance = pd.DataFrame({
        "feature": columns,
        "importance": importance
    })

    # sort
    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # vis
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance,
    )
    plt.title(f"Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))


if __name__ == "__main__":
    main()
