#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/pvhm256/ \
    --dataset_mode aligned_with_C\
    --name ftae_flow_rect_grid_tv_pvhm\
    --model ftae_flow \
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
    --nz 200\
    --rectified\
    --add_grid\
    --lambda_tv 1\


#        --which_direction BtoA\




#    --add_grid




