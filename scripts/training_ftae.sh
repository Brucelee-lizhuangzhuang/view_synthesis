#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/PVHM_dataset \
    --dataset_mode aligned\
    --name ftae\
    --model ftae \
    --identity 0 \
    --save_epoch_freq 50 \
    --which_model_netD n_layers \
    --n_layers_D 2 \
    --loadSize 128 \
    --fineSize 128 \
    --no_flip\
    --display_freq 100 \
    --display_port 8099 \
    --no_dropout \
    --lr 0.0001 \
    --niter 100 \
    --niter_decay 100 \
    --lambda_gan 0 \
    --batchSize 16 \
    --nz 200




