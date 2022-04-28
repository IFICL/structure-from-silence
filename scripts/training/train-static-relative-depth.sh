#!/bin/bash

source ~/.bashrc
conda activate SFS

# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp='Staic-relative-depth' --epochs=50  \
    --setting='static_relative_depth' --input='audio' --backbone='vggish' \
    --batch_size=128 --num_workers=4 --save_step=5 --valid_step=1 --lr=0.001 \
    --optim='SGD' --repeat=1 --schedule='cos'