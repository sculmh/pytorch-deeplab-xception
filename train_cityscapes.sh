#!/usr/bin/env bash

# experiment 1
# python train.py --backbone xception --lr 0.01 --workers 4 --epochs 200 --batch-size 4 --test-batch-size 4 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes

# experiment 2
# python train.py --backbone xception --lr 0.01 --workers 4 --epochs 200 --batch-size 2 --test-batch-size 2 --base-size 769 --crop-size 769 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes

# experiment 3
# python train.py --backbone xception --lr 0.001 --workers 4 --epochs 1599 --batch-size 4 --test-batch-size 4 --groups 32 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes

# experiment 4
# python train.py --backbone xception --lr 0.001 --workers 4 --epochs 1599 --batch-size 4 --test-batch-size 4 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes

# experiment 5
# python train.py --backbone xception --lr 0.001 --workers 4 --epochs 800 --batch-size 2 --test-batch-size 2 --groups 32 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes

# experiment 6
# python train.py --backbone xception --lr 0.001 --workers 4 --epochs 800 --batch-size 2 --test-batch-size 2 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes