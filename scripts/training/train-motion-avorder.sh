#!/bin/bash

source ~/.bashrc
conda activate SFS

# running jobs
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp='Motion-AVorder' --epochs=30  \
    --setting='motion_avorder' --input='both' --backbone='vggish' \
    --batch_size=128 --num_workers=8 --save_step=1 --valid_step=1 --lr=0.0001 \
    --optim='Adam' --repeat=50 --schedule='cos' --aug_wave --pretrained