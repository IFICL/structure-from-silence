#!/bin/bash

source ~/.bashrc
conda activate SFS

# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp='Motion-obstacle-detection' --epochs=40  \
    --setting='motion_obstacle' --input='audio' --backbone='vggish' \
    --batch_size=256 --num_workers=4 --save_step=2 --valid_step=1 --lr=0.001 \
    --optim='SGD' --repeat=1 --schedule='cos'