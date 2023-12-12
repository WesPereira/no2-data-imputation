#!/bin/bash

for kernel_size in 5 7 9 11
do
    for nl in 1 2 3
    do
        for hd in 16 32 64
        do
            for bs in 16 32 64 128
            do
                for lr in 1e-3 5e-3 1e-4 5e-4
                do
                    python train_gru.py convlstm --lr=$lr --batch_size=$bs --hidden_dim=$hd --nl=$nl --kernel_size=$kernel_size --convs="[8, 4, 2]"
                done
            done
        done
    done
done