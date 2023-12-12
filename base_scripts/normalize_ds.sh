#!/bin/bash

cd src/data
python fill_npz.py --mapped_path="../../data/interim/gee_ds_20231003_temporal_drift_analysis/mapped_paths.csv" \
                   --artifacts_path="../../artifacts/gee_ds_20231003_temporal_drift_analysis"
#                   --normalizer="../../artifacts/121023/standard_scaler.joblib"
