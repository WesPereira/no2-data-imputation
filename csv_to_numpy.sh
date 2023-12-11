#!/bin/bash

cd src/data
python csv2numpy.py --data_path="../../data/raw/gee_ds_20231204_2" \
                    --output_path="../../data/interim/gee_ds_20231204_2"
cd ../..
