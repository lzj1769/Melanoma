#!/usr/local_rwth/bin/zsh

module load cuda
source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

export CUDA_VISIBLE_DEVICES=0,1

python train.py \
--arch $1 \
--image_width $2 \
--image_height $3 \
--per_gpu_batch_size $4 \
--fold $5 \
--pretrained \
--log