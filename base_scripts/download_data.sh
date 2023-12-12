#!/bin/bash

python download_data.py --gee_dataset="NASA/SMAP/SPL4SMGP/007" \
                        --dataset_bands=["sm_surface"] \
                        --list_of_geoms="points.json" \
                        --output_path="sm_surface.csv"