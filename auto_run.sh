#!/usr/bin/env bash

set -e  # 오류 발생 시 즉시 중단하도록 설정

# python train.py --student_frame_type embedding_mix_raw --student_model_type XGBoost --embedding_type BERT
# python train.py --student_frame_type embedding_mix_raw --student_model_type LightGBM --embedding_type BERT
# python train.py --student_frame_type embedding_mix_raw --student_model_type MLP --embedding_type BERT
# python train.py --student_frame_type embedding_mix_raw --student_model_type ResidualMLP --embedding_type BERT
# python train.py --student_frame_type embedding_mix_raw --student_model_type TabNet --embedding_type BERT

# python train.py --student_frame_type embedding_raw --student_model_type XGBoost --embedding_type BERT
# python train.py --student_frame_type embedding_raw --student_model_type LightGBM --embedding_type BERT
# python train.py --student_frame_type embedding_raw --student_model_type TabNet --embedding_type BERT


# python train.py --student_frame_type embedding_raw --student_model_type MLP --embedding_type GPT2
# python train.py --student_frame_type embedding_raw --student_model_type ResidualMLP --embedding_type GPT2
# python train.py --student_frame_type embedding_raw --student_model_type XGBoost --embedding_type GPT2
# python train.py --student_frame_type embedding_raw --student_model_type LightGBM --embedding_type GPT2
# python train.py --student_frame_type embedding_raw --student_model_type TabNet --embedding_type GPT2






# python train.py --student_frame_type embedding_raw --student_model_type MLP --embedding_type GEMMA3
# python train.py --student_frame_type embedding_raw --student_model_type ResidualMLP --embedding_type GEMMA3






# python train.py --student_frame_type embedding_raw --student_model_type XGBoost --embedding_type GEMMA3
# python train.py --student_frame_type embedding_raw --student_model_type LightGBM --embedding_type GEMMA3
# python train.py --student_frame_type embedding_raw --student_model_type TabNet --embedding_type GEMMA3



python train.py --student_frame_type embedding_raw --student_model_type MLP --embedding_type ELECTRA
python train.py --student_frame_type embedding_raw --student_model_type ResidualMLP --embedding_type ELECTRA
# python train.py --student_frame_type embedding_raw --student_model_type XGBoost --embedding_type ELECTRA
# python train.py --student_frame_type embedding_raw --student_model_type LightGBM --embedding_type ELECTRA
# python train.py --student_frame_type embedding_raw --student_model_type TabNet --embedding_type ELECTRA


python train.py --student_frame_type embedding_raw --student_model_type MLP --embedding_type T5
python train.py --student_frame_type embedding_raw --student_model_type ResidualMLP --embedding_type T5
# python train.py --student_frame_type embedding_raw --student_model_type XGBoost --embedding_type T5
# python train.py --student_frame_type embedding_raw --student_model_type LightGBM --embedding_type T5
# python train.py --student_frame_type embedding_raw --student_model_type TabNet --embedding_type T5



python train.py --student_frame_type embedding_mix_raw --student_model_type XGBoost --embedding_type GPT2
# python train.py --student_frame_type embedding_mix_raw --student_model_type LightGBM --embedding_type GPT2
python train.py --student_frame_type embedding_mix_raw --student_model_type MLP --embedding_type GPT2
python train.py --student_frame_type embedding_mix_raw --student_model_type ResidualMLP --embedding_type GPT2
# python train.py --student_frame_type embedding_mix_raw --student_model_type TabNet --embedding_type GPT2


python train.py --student_frame_type embedding_mix_raw --student_model_type XGBoost --embedding_type GEMMA3
# python train.py --student_frame_type embedding_mix_raw --student_model_type LightGBM --embedding_type GEMMA3
python train.py --student_frame_type embedding_mix_raw --student_model_type MLP --embedding_type GEMMA3
python train.py --student_frame_type embedding_mix_raw --student_model_type ResidualMLP --embedding_type GEMMA3
# python train.py --student_frame_type embedding_mix_raw --student_model_type TabNet --embedding_type GEMMA3



python train.py --student_frame_type embedding_mix_raw --student_model_type XGBoost --embedding_type ELECTRA
# python train.py --student_frame_type embedding_mix_raw --student_model_type LightGBM --embedding_type ELECTRA
python train.py --student_frame_type embedding_mix_raw --student_model_type MLP --embedding_type ELECTRA
python train.py --student_frame_type embedding_mix_raw --student_model_type ResidualMLP --embedding_type ELECTRA
# python train.py --student_frame_type embedding_mix_raw --student_model_type TabNet --embedding_type ELECTRA


python train.py --student_frame_type embedding_mix_raw --student_model_type XGBoost --embedding_type T5
# python train.py --student_frame_type embedding_mix_raw --student_model_type LightGBM --embedding_type T5
python train.py --student_frame_type embedding_mix_raw --student_model_type MLP --embedding_type T5
python train.py --student_frame_type embedding_mix_raw --student_model_type ResidualMLP --embedding_type T5
# python train.py --student_frame_type embedding_mix_raw --student_model_type TabNet --embedding_type T5
