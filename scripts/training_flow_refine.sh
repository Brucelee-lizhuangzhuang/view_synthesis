#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/pvhm_test/ \
    --dataset_mode aligned_with_C\
    --name flow_refine\
    --model flow_refine \
    --identity 0 \
    --save_epoch_freq 100 \
    --which_model_netD n_layers \
    --n_layers_D 2\
    --which_model_netG unet_128\
    --loadSize 128\
    --fineSize 128\
    --no_flip\
    --display_freq 100\
    --ngf 64\
    --display_port 8098\
    --no_dropout\
    --lr 0.0002\
    --which_direction BtoA\
    --lambda_gan 0




