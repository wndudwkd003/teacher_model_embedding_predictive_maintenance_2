import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config.configs import Config, DataType, ModelType, RunMode
from models.data_module import DataFactory
from models.factories import TrainerFactory


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(config: Config):
    """
    # 계획
    1. 딥러닝, 머신러닝 비교 모델로 전부 학습해야 함
    2. 시나리오별로 학습하고 평가도 시나리오별로 진행
    3. 잘된 시나리오 모델을 최종 선정
    4. 평가 기준 적분 이용하는 방법 구상
    5. 교사-학생 모델 조합 각각 진행
    6. 한번에 학습 하는 구조가 아니라 Config에서 선택할 수 있도록 구성

    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # train
    if config.run_mode in [RunMode.TRAIN, RunMode.TRAIN_TEST]:
        print("[INFO] Train Mode")
        train_save_path = get_save_path(config.train_valid_result_path, config.student_data_type, config.student_data_type, current_time)
        reports = []
        # train by scenario
        for scen in config.train_scenario:
            print(f"[INFO] Current Scenario: {scen}")
            report = train_pipe(scen, config, train_save_path)
            report["scenario"] = scen
            print("accuracy: ", report["accuracy"])
            reports.append(report)

        # valid report save
        report_df = pd.DataFrame(reports)
        report_df.to_csv(os.path.join(train_save_path, "report.csv"), index=False)
        print(f"[INFO] Train reports saved to {train_save_path}")

    # test
    if config.run_mode in [RunMode.TEST, RunMode.TRAIN_TEST]:
        print("[INFO] Test Mode")
        if config.run_mode == RunMode.TRAIN_TEST:
            test_save_path = train_save_path
        else:
            test_save_path = get_save_path(config.test_result_path, config.student_data_type, config.student_data_type, current_time)
        model_scens_reports = []
        # test by scenario
        for scen in config.train_scenario:
            print(f"[INFO] Current Scenario: {scen}")
            reports = test_pipe(scen, config)
            model_scens_reports.append(reports)

        # test report save
        svae_test_reports(model_scens_reports, config, test_save_path)
        print(f"[INFO] Test reports saved to {test_save_path}")



def vis_graph(json_path: str, metrics: list = ["accuracy", "f1-score"]):
    pass


def svae_test_reports(reports: list, config: Config, save_path: str):
    rows = []
    for model_scen, scen_reports in zip(config.train_scenario, reports):
        for rep in scen_reports:
            rep["model_scenario"] = model_scen
            rows.append(rep)
    df = pd.DataFrame(rows)
    json_path = os.path.join(save_path, "test_report.json")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    # vis_graph(json_path)


def test_pipe(scen: int, config: Config):
    """
    # 계획
    1. 모델은 시나리오별로 학습되어있음. 14개 각각의 모델을 모든 시나리오로 평가해야 함
    2. 상위 반복문에서 시나리오 반복으로 현재 시나리오로 학습된 모델을 현재 함수에서 불러옴
    3. 모델을 모든 시나리오 평가 데이터 세트로 평가 진행

    """

    # model load
    trainer = TrainerFactory.build(config.student_model_type, config)
    trainer.load_model(os.path.join(config.test_model_save_path, "model_save"), f"scen_{scen}")

    # data load
    scen_reports = []
    for i in config.train_scenario:
        X_valid, y_valid = DataFactory.load(config.test_data_path, "valid", i, DataType.RAW)
        if config.student_data_type != DataType.RAW:
            X_valid, _ = DataFactory.load(config.test_data_path, "valid", i, config.embedding_type)

        report = trainer.eval(X_valid, y_valid, digits=5, output_dict=True)
        report["scenario"] = i
        scen_reports.append(report)

    return scen_reports


def train_pipe(scen: int, config: Config, save_path: str):
    # data laod todo
    X_train, y_train = DataFactory.load(config.train_data_path, "train", scen, DataType.RAW)
    X_valid, y_valid = DataFactory.load(config.train_data_path, "valid", scen, DataType.RAW)

    if config.student_data_type != DataType.RAW:
        X_train, _ = DataFactory.load(config.train_data_path, "train", scen, config.embedding_type)
        X_valid, _ = DataFactory.load(config.train_data_path, "valid", scen, config.embedding_type)

    # model trainer
    trainer = TrainerFactory.build(config.student_model_type, config)
    trainer.fit(X_train, y_train, X_valid, y_valid, verbose=False)
    report = trainer.eval(X_valid, y_valid, digits=5, output_dict=True)

    # save path
    model_save_path = os.path.join(save_path, "model_save")
    os.makedirs(model_save_path, exist_ok=True)

    # model save
    trainer.save_model(model_save_path, f"scen_{scen}")

    return report

def get_save_path(result_path: str, data_type: DataType, model_type: ModelType, current_time: str):
    save_path = os.path.join(result_path, f"{data_type.name}_{model_type.value}_{current_time}")
    os.makedirs(save_path, exist_ok=True)
    return save_path

if __name__ == "__main__":
    config = Config()
    seed_everything(config.seed)
    main(config)
