#!/bin/bash

declare -a arr=("[8, 2, 1]" "[8, 3, 1]")

for bs in 32 64 128
do
    for lr in 1e-2 3 e-2 5e-2 1e-1 5e-3
    do
        for kernel_size in 23 19 15 11 7
        do
            for convs in "${arr[@]}"
            do
                python train_gru.py conv1d --lr=$lr --batch_size=$bs --convs="${convs}" --kernel_size=$kernel_size
            done
        done
    done
done