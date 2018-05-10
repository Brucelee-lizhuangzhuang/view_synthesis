#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/car/ \
    --dataset_mode aligned\
    --name car64\
    --model ftae_flow \
    --identity 0 \
    --save_epoch_freq 50 \
    --which_model_netD n_layers \
    --n_layers_D 2 \
    --loadSize 64 \
    --fineSize 64 \
    --no_flip\
    --display_freq 100 \
    --display_port 8098 \
    --no_dropout \
    --lr 0.00006 \
    --niter 100 \
    --lambda_gan 0 \
    --batchSize 16 \
    --nz 200\
    --lambda_tv 1\
    --lambda_flow 0\
    --norm batch\


#    --add_grid\

#    --upsample bilinear

#    --niter_decay 100 \

    #--which_direction BtoA\




#    --add_grid




