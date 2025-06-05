#!/usr/bin/env bash

set -e  # 오류 발생 시 즉시 중단하도록 설정

python train.py \
    --is_kd_mode true \
    --teacher_model_type ResidualMLP \
    --teacher_frame_type raw_raw \
    --teacher_data_type RAW \
    --teacher_model_save_path "path/to/your/pretrained_teacher_resnet.pth" \
    --student_model_type MLP \
    --student_frame_type embedding_logits \
    --student_data_type BERT
