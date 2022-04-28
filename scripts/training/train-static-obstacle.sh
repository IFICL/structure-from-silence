#!/bin/bash

source ~/.bashrc
conda activate SFS

# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp='Staic-obstacle-detection' --epochs=100  \
    --setting='static_obstacle' --input='audio' --backbone='vggish' \
    --batch_size=256 --num_workers=4 --save_step=5 --valid_step=1 --lr=0.001 \
    --optim='SGD' --repeat=10 --schedule='cos' --aug_wave