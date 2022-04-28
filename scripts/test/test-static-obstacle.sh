#!/bin/bash

source ~/.bashrc
conda activate SFS

Model='pretrained-models/static-obstacle-detection.pth.tar'
# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp='Staic-obstacle-detection' \
    --setting='static_obstacle' --input='audio' --backbone='vggish' \
    --batch_size=256 --num_workers=4 --test_mode --resume=$Model