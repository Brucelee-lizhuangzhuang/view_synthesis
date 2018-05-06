#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/car_multi_view/ \
    --dataset_mode aligned_multi_view\
    --name car_multi_view_depth_ftae_vae\
    --model multi_view_depth \
    --which_model_netG unet_128\
    --save_epoch_freq 50 \
    --loadSize 128 \
    --fineSize 128 \
    --no_flip\
    --display_freq 100 \
    --display_port 8098 \
    --no_dropout \
    --lr 0.00006 \
    --niter 200 \
    --niter_decay 200 \
    --lambda_gan 0 \
    --batchSize 16 \
    --nz 200\
    --lambda_tv 0\
    --lambda_flow 0\
    --norm batch\
    --ignore_center\
    --use_vae\
    --lambda_kl 0.01

#    --random_AB

#    --concat_grid\

#    --nl_dec relu

#    --use_pyramid\

#    --ignore_center

#    --lambda_kl 0.01

#    --add_grid\

#    --upsample bilinear


    #--which_direction BtoA\




#    --add_grid




