#!/bin/bash

#python get_metrics_n_plots.py --infer_path="artifacts/models_05112023/gru/lr0.01_bs32_hd32_nl1/290543391431972567.csv" \
#                              --output_folder="plots/final/gru"

#python get_metrics_n_plots.py --infer_path="artifacts/models_05112023/lstm/lr0.01_bs128_hd32_nl1/840623644518907001.csv" \
#                              --output_folder="plots/final/lstm"

#python get_metrics_n_plots.py --infer_path="artifacts/models_05112023/convlstm/lr0.0001_bs32_kz_5_nl1_hd16_convs8_4_2/608931432535340336.csv" \
#                              --output_folder="plots/final/convlstm"

python get_metrics_n_plots.py --infer_path="artifacts/models_05112023/conv1d/lr0.001_bs32_lins321_32_1_convs8_2_1/118561025684032372.csv" \
                              --output_folder="plots/final/conv1d"