#!/bin/bash
dataset=$1
python train.py\
    --dataroot ~/data/$dataset \
    --dataset_mode aligned\
    --name ${dataset}_pix2pix\
    --model pix2pix \
    --identity 0 \
    --save_epoch_freq 100 \
    --which_model_netD n_layers \
    --n_layers_D 2\
    --which_model_netG resnet_6blocks\
    --loadSize 128\
    --fineSize 128\
    --no_flip\
    --display_freq 10\
    --ngf 64\
    --display_port 8099\
    --no_dropout\
    --lr 0.00006\




