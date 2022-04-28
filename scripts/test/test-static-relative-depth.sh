#!/bin/bash

source ~/.bashrc
conda activate SFS

Model='pretrained-models/static-relative-depth.pth.tar'

# running jobs
CUDA_VISIBLE_DEVICES=0 python main.py --exp='Staic-relative-depth'  \
    --setting='static_relative_depth' --input='audio' --backbone='vggish' \
    --batch_size=128 --num_workers=4 --test_mode --resume=$Model