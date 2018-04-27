#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/pvhm256/ \
    --dataset_mode aligned_with_C\
    --name ftae_flow_256\
    --model ftae_flow \
    --identity 0 \
    --no_flip\
    --no_dropout \
    --save_epoch_freq 50 \
    --display_freq 100 \
    --display_port 8099 \
    --loadSize 256 \
    --fineSize 256 \
    --lr 0.00006 \
    --niter 100 \
    --niter_decay 100 \
    --batchSize 16 \
    --nz 400\
    --lambda_tv 0 \
    --lambda_gan 0 \
    --lambda_flow 0 \
    --add_grid\
    --rectified\

#    --upsample bilinear



#        --which_direction BtoA\
#    --add_grid




