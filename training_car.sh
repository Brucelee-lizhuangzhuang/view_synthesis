#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/car_v1/ \
    --dataset_mode aligned_multi_view_random\
    --name car_mask \
    --model multi_view_depth_random \
    --save_epoch_freq 50 \
    --loadSize 256 \
    --fineSize 256 \
    --no_flip\
    --display_freq 100 \
    --display_port 8098 \
    --no_dropout \
    --lr 0.0002 \
    --niter 400 \
    --niter_decay 400 \
    --lambda_gan 0 \
    --batchSize 16 \
    --nz 200\
    --lambda_tv 1\
    --norm batch\
    --ignore_center\
    --train_split 1\
    --number_samples 2\
    --category car1\

#    --pred_mask

#    --only_neighbour


    #--only_neighbour

# 0.00002
    #    --use_pyramid\

#    --n_bilinear_layers 3

#    --use_vae\
#    --lambda_kl 0.01

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




