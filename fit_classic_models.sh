#!/bin/bash

python fit_classic_models.py --ds_path="data/processed/gee_ds_20231003_2022_test" \
                      --pca_path="artifacts/121023_2022_test/pca_550_components.joblib" \
                      --out_path="artifacts/121023_2022_test"
