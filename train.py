import os
import random
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config.configs import Config, DataType, ModelType, RunType, StackType, FrameType
from models.data_module import DataFactory, DataManager
from models.factories import TrainerFactory

from utils.config_to_dict import config_to_dict
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from argparse import ArgumentParser

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
    if config.run_mode in [RunType.TRAIN, RunType.TRAIN_TEST]:
        print("[INFO] Train Mode")
        train_save_path = get_save_path(config, current_time)

        config_dict = config_to_dict(config)
        config_json_path = os.path.join(train_save_path, "config.json")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Config saved to {config_json_path}")


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
    if config.run_mode in [RunType.TEST, RunType.TRAIN_TEST]:
        print("[INFO] Test Mode")
        if config.run_mode == RunType.TRAIN_TEST:
            model_path = train_save_path
            save_dir = os.path.basename(model_path)
        else:
            model_path = config.test_model_save_path
            save_dir = os.path.basename(model_path)
        model_scens_reports = []
        # test by scenario
        print(f"[INFO] Model Path: {model_path}")
        for scen in config.train_scenario:
            print(f"[INFO] Current Scenario: {scen}")
            reports = test_pipe(scen, config, model_path)
            print_each_scnario(scen, reports)
            model_scens_reports.append(reports)

        # test report save
        save_dir = os.path.join(config.test_result_path, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        svae_test_reports(model_scens_reports, config, save_dir)
        print(f"[INFO] Test reports saved to {save_dir}")

def print_each_scnario(scen: int, reports: list):
    for i, report in enumerate(reports):
        print(f"Scene {i} -> ACC: {report['accuracy']} F1: {report['macro avg']['f1-score']} Precision: {report['macro avg']['precision']} Recall: {report['macro avg']['recall']}")
    print()

def vis_graph(json_path: str, metrics: list = ["accuracy", "f1_score"]):
    df = pd.read_json(json_path, orient="records", convert_dates=False)

    long_df = df.melt(
        id_vars=["model_scenario", "scenario"],
        value_vars=metrics,
        var_name="metric",
        value_name="score"
    )

    palette = sns.color_palette("tab20", long_df["model_scenario"].nunique())

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=long_df,
        x="scenario",
        y="score",
        hue="model_scenario",
        style="metric",
        markers=True,
        dashes=False,
        palette=palette
    )

    plt.title("accuracy & f1_score by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Score")

    # 범례를 그림 영역 밖 오른쪽 상단에 배치
    plt.legend(title="Model Scenario / Metric", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    img_path = json_path.replace(".json", ".jpg")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()


def svae_test_reports(reports: list, config: Config, save_dir: str):
    nested = {}
    flat_rows = []

    for model_scen, scen_reports in zip(config.train_scenario, reports):
        m_key = str(model_scen)
        nested[m_key] = {}
        for rep in scen_reports:
            s_key = str(rep["scenario"])
            metrics_dict = {
                "accuracy": rep["accuracy"],
                "f1_score": rep["macro avg"]["f1-score"],
                "precision": rep["macro avg"]["precision"],
                "recall": rep["macro avg"]["recall"]
            }
            nested[m_key][s_key] = metrics_dict

            # 그래프용 평탄화
            flat_rows.append({
                "model_scenario": model_scen,
                "scenario": rep["scenario"],
                **metrics_dict
            })

    os.makedirs(save_dir, exist_ok=True)
    nested_path = os.path.join(save_dir, "test_report_nested.json")
    with open(nested_path, "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)


    flat_df = pd.DataFrame(flat_rows)
    flat_path = os.path.join(save_dir, "test_report.json")
    flat_df.to_json(flat_path, orient="records", force_ascii=False, indent=2)

    # 필요 지표만 전달
    vis_graph(flat_path, metrics=["accuracy", "f1_score"])



def test_pipe(scen: int, config: Config, model_path: str):
    """
    # 계획
    1. 모델은 시나리오별로 학습되어있음. 14개 각각의 모델을 모든 시나리오로 평가해야 함
    2. 상위 반복문에서 시나리오 반복으로 현재 시나리오로 학습된 모델을 현재 함수에서 불러옴
    3. 모델을 모든 시나리오 평가 데이터 세트로 평가 진행

    """

    # test pipe는 masking 데이터 세트로 진행
    current_pipe_path = config.test_data_path

    # model load
    trainer = TrainerFactory.build(config.student_model_type, config)
    trainer.load_model(os.path.join(model_path, "model_save"), f"scen_{scen}")

    # data load
    scen_reports = []
    for current_scen_for_test in config.train_scenario:
        print(f"[INFO] Current Scenario for Test: {current_scen_for_test}")
        if config.is_kd_mode:
            teacher_X_valid, student_X_valid, y_valid = DataManager.get_masking_test_data(current_pipe_path, config, current_scen_for_test)
            print(f"[INFO] KD Mode Valid Shapes | Teacher: {teacher_X_valid.shape}, Student: {student_X_valid.shape}, y: {y_valid.shape}")

            report = trainer.eval(
                student_X_valid, y_valid,
                digits=5,
                output_dict=True
            )

        else:
            X_valid, y_valid = DataManager.get_masking_test_data(current_pipe_path, config, current_scen_for_test)
            print(f"[INFO] Standard Mode Valid Shape: {X_valid.shape}, {y_valid.shape}")

            report = trainer.eval(
                X_valid, y_valid,
                digits=5,
                output_dict=True
            )

        report["scenario"] = current_scen_for_test
        scen_reports.append(report)

    return scen_reports


def train_pipe(scen: int, config: Config, save_path: str):
    # train pipe는 일반 split 데이터 세트로 진행
    current_pipe_path = config.train_data_path

    # data laod todo
    if config.is_kd_mode:
        (teacher_X_train, student_X_train, y_train), \
        (teacher_X_valid, student_X_valid, y_valid) = DataManager.get_train_data(current_pipe_path, config, scen)
        print(f"[INFO] KD Mode Train Shapes | Teacher: {teacher_X_train.shape}, Student: {student_X_train.shape}, y: {y_train.shape}")
        print(f"[INFO] KD Mode Valid Shapes | Teacher: {teacher_X_valid.shape}, Student: {student_X_valid.shape}, y: {y_valid.shape}")

    else:
        (X_train, y_train), (X_valid, y_valid) = DataManager.get_train_data(current_pipe_path, config, scen)
        print(f"[INFO] Standard Mode Train Shape: {X_train.shape}, {y_train.shape}")
        print(f"[INFO] Standard Mode Valid Shape: {X_valid.shape}, {y_valid.shape}")


    # model trainer
    if config.is_kd_mode:
        trainer = TrainerFactory.build_distill_trainer(
            config.student_model_type,
            config.teacher_model_type,
            config,
            scen
        )
    else:
        trainer = TrainerFactory.build(config.student_model_type, config)

    if config.is_kd_mode:
        # train
        trainer.fit(
            teacher_X_train, student_X_train, y_train,
            teacher_X_valid, student_X_valid, y_valid,
            verbose=False
        )
        # evaluation
        report = trainer.eval(
            student_X_valid, y_valid,
            digits=5,
            output_dict=True)
    else:
        # train
        trainer.fit(
            X_train, y_train,
            X_valid, y_valid,
            verbose=False
        )
        # evaluation
        report = trainer.eval(
            X_valid, y_valid,
            digits=5,
            output_dict=True
        )

    # save path
    model_save_path = os.path.join(save_path, "model_save")
    os.makedirs(model_save_path, exist_ok=True)

    # model save
    trainer.save_model(model_save_path, f"scen_{scen}")

    return report

def get_save_path(config: Config, current_time: str):
    result_path = config.train_valid_result_path

    if config.is_kd_mode:
        teacher_frame_type = config.teacher_frame_type
        teacher_model_type = config.teacher_model_type
        teacher_data_type = config.teacher_data_type
        student_frame_type = config.student_frame_type
        student_model_type = config.student_model_type
        student_data_type = config.student_data_type
        save_path = os.path.join(
            result_path,
            f"{student_frame_type.name}_{student_model_type.value}_{student_data_type.name}_teacher_"
            f"{teacher_frame_type.name}_{teacher_model_type.value}_{teacher_data_type.name}_"
            f"{current_time}"
        )

    else:
        student_frame_type = config.student_frame_type
        student_model_type = config.student_model_type
        student_data_type = config.student_data_type
        save_path = os.path.join(
            result_path,
            f"{student_frame_type.name}_{student_model_type.value}_{student_data_type.name}_{current_time}"
        )

    os.makedirs(save_path, exist_ok=True)
    return save_path

if __name__ == "__main__":
    config = Config()
    parser = ArgumentParser()

    # --- 기존 학생 모델 설정 인자 ---
    parser.add_argument(
        "--student_frame_type", type=str, choices=[e.value for e in FrameType],
        default=config.student_frame_type.value, help="학생 모델 FrameType"
    )
    parser.add_argument(
        "--student_model_type", type=str, choices=[e.value for e in ModelType],
        default=config.student_model_type.value, help="학생 모델 ModelType"
    )
    parser.add_argument(
        "--student_data_type", type=str, choices=[e.value for e in DataType],
        default=config.student_data_type.value, help="학생 모델 임베딩 DataType"
    )

    # --- ✨ 교사 모델 및 KD 모드 설정 인자 추가 ---
    parser.add_argument(
        "--is_kd_mode", type=lambda x: (str(x).lower() == 'true'), # 문자열 'true'/'false'를 bool로 변환
        default=config.is_kd_mode, help="지식 증류 모드 사용 여부 (True/False)"
    )
    parser.add_argument(
        "--teacher_frame_type", type=str, choices=[e.value for e in FrameType],
        default=config.teacher_frame_type.value, help="교사 모델 FrameType"
    )
    parser.add_argument(
        "--teacher_model_type", type=str, choices=[e.value for e in ModelType],
        default=config.teacher_model_type.value, help="교사 모델 ModelType"
    )
    parser.add_argument(
        "--teacher_data_type", type=str, choices=[e.value for e in DataType],
        default=config.teacher_data_type.value, help="교사 모델 DataType"
    )
    parser.add_argument(
        "--teacher_model_save_path", type=str,
        default=config.teacher_model_save_path, help="사전 학습된 교사 모델 경로"
    )
    # 필요한 경우, 다른 Config 변수들도 위와 같이 추가할 수 있습니다. (예: --lr, --batch_size 등)

    args = parser.parse_args()

    # Config 객체 업데이트
    config.student_frame_type = FrameType(args.student_frame_type)
    config.student_model_type = ModelType(args.student_model_type)
    config.student_data_type = DataType(args.student_data_type)

    config.is_kd_mode = args.is_kd_mode
    config.teacher_frame_type = FrameType(args.teacher_frame_type)
    config.teacher_model_type = ModelType(args.teacher_model_type)
    config.teacher_data_type = DataType(args.teacher_data_type)
    config.teacher_model_save_path = args.teacher_model_save_path

    seed_everything(config.seed)
    print(config) # 실행 전 최종 config 확인
    main(config)
