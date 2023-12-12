#!/bin/bash

cd src/data
python format_csvs.py --data_path="../../data/raw/gee_ds_20231204" \
                      --output_path="../../data/raw/gee_ds_20231204"
cd ../..
