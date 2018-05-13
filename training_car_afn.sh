#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/car_v1/ \
    --dataset_mode appearance_flow\
    --name car_afn \
    --model appearance_flow \
    --save_epoch_freq 50 \
    --loadSize 256 \
    --fineSize 256 \
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
    --lambda_tv 0\
    --norm batch\
    --train_split 1\
    --category car1\
    --nl_enc relu\
    --only_neighbour

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




