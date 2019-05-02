#!/usr/bin/env bash

python train.py --backbone xception --lr 0.01 --workers 4 --epochs 200 --batch-size 4 --test-batch-size 4 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname deeplab-xception --eval-interval 1 --dataset cityscapes