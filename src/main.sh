#!/bin/bash

# 设置默认参数 
PHASE="train_step1"  #取值有： train_step1 train_step2_pred_gent train_step2 train_step3 test  其中：train_step1最原始的，train_step2保存到外部的pred，再加载合并；train_step3在模型里合并pred。
BATCH_SIZE=1
INPUT_SIZE=96
INPUT_CHANNELS=1
OUTPUT_SIZE=96
OUTPUT_CHANNELS=9
PRED_FILTER="1,2,6"
RENAME_MAP="0, 1, 2, 3, 4, 5, 6, 7, 8"
RESIZE_RATIO=0.9
TRAIN_DATA_DIR="../../../HOCM24/original"
PRED_LABELING_DIR="../../../HOCM24/train/pred_label"
CHECKPOINT_DIR="../outcome/model/checkpoint"
CHECKPOINT_DIR2="../outcome/model/checkpoint2"
CHECKPOINT_DIR3="../outcome/model/checkpoint3"
LEARNING_RATE=0.001
BETA1=0.5
EPOCHS=54000
MODEL_NAME="HOCM-Net.model"
SAVE_INTERVAL=2000
TEST_DATA_DIR="../../../HOCM24/test/image"
LABELING_DIR="../result"
TEST_LABEL_DIR="../../../HOCM24/test/label"
OVERLAP_ITER=4

# 将参数传递给 Python 脚本
python -B main.py \
  --phase "$PHASE" \
  --batch_size "$BATCH_SIZE" \
  --inputI_size "$INPUT_SIZE" \
  --inputI_chn "$INPUT_CHANNELS" \
  --outputI_size "$OUTPUT_SIZE" \
  --output_chn "$OUTPUT_CHANNELS" \
  --pred_filter "$PRED_FILTER" \
  --rename_map "$RENAME_MAP" \
  --resize_r "$RESIZE_RATIO" \
  --traindata_dir "$TRAIN_DATA_DIR" \
  --chkpoint_dir "$CHECKPOINT_DIR" \
  --chkpoint_dir2 "$CHECKPOINT_DIR2" \
  --chkpoint_dir3 "$CHECKPOINT_DIR3" \
  --learning_rate "$LEARNING_RATE" \
  --beta1 "$BETA1" \
  --epoch "$EPOCHS" \
  --model_name "$MODEL_NAME" \
  --save_intval "$SAVE_INTERVAL" \
  --testdata_dir "$TEST_DATA_DIR" \
  --labeling_dir "$LABELING_DIR" \
  --testlabel_dir "$TEST_LABEL_DIR" \
  --predlabel_dir "$PRED_LABELING_DIR" \
  --ovlp_ita "$OVERLAP_ITER"
