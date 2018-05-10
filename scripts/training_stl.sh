#!/bin/bash
dataset=$1
python train.py\
    --dataroot /home/xu/data/view_synthesis/frontal2side \
    --dataset_mode aligned_with_flow\
    --name stl_flow\
    --model pix2pix_stl \
    --identity 0 \
    --save_epoch_freq 50 \
    --which_model_netD n_layers \
    --n_layers_D 2 \
    --which_model_netG resnet_6blocks\
    --loadSize 128 \
    --fineSize 128 \
    --no_flip\
    --display_freq 100 \
    --ngf 64 \
    --display_port 8099 \
    --no_dropout \
    --lr 0.000006 \
    --niter 100 \
    --niter_decay 100 \
    --lambda_gan 0 \
    --lambda_flow 10\
    --lambda_A 10\
    --which_direction BtoA




