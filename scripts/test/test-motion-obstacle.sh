#!/bin/bash

source ~/.bashrc
conda activate SFS

Model='pretrained-models/motion-obstacle-detection.pth.tar'

# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp='Motion-obstacle-detection'  \
    --setting='motion_obstacle' --input='audio' --backbone='vggish' \
    --batch_size=256 --num_workers=4 --test_mode --resume=$Model