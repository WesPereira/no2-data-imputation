#!/bin/bash

python get_infer.py --model_path="model_31-07-2023/lr0.01_bs32_hd32_nl1/epoch=14-val_loss=295144.0938-val_r2=0.5386.ckpt" \
                    --test_path="data/processed/dataset_max/test_df_mapping.csv" \
                    --output_path="lr0.01_bs32_hd32_nl1.csv"