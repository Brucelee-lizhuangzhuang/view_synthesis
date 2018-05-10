#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/pvhm256/ \
    --dataset_mode aligned_with_C\
    --name debug\
    --model ftae_flow \
    --identity 0 \
    --save_epoch_freq 50 \
    --which_model_netD n_layers \
    --n_layers_D 2 \
    --loadSize 128 \
    --fineSize 128 \
    --no_flip\
    --display_freq 100 \
    --display_port 8098 \
    --no_dropout \
    --lr 0.00002 \
    --niter 100 \
    --lambda_gan 0 \
    --batchSize 16 \
    --nz 200\
    --rectified\
    --add_grid\
    --lambda_tv 1\
    --lambda_flow 10\
    --norm batch

#    --niter_decay 100 \

    #--which_direction BtoA\




#    --add_grid




