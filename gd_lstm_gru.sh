#!/bin/bash

for bs in 32 64 128
do
    for lr in 1e-1 2e-1 5e-2 1e-2
    do
        for hd in 8 16 32 64
        do
            for nl in 1 2
            do
                python train_gru.py lstm_n_gru --lr=$lr --batch_size=$bs --hidden_dim=$hd --nl=$nl
            done
        done
    done
done