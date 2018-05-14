#!/bin/bash
python train.py\
    --dataroot ~/data/view_synthesis/human/ \
    --dataset_mode aligned_multi_view_random\
    --name human_full_random \
    --model multi_view_depth_random \
    --save_epoch_freq 50 \
    --loadSize 128 \
    --fineSize 128 \
    --no_flip\
    --display_freq 100 \
    --display_port 8099 \
    --no_dropout \
    --lr 0.00002 \
    --niter 400 \
    --niter_decay 400 \
    --lambda_gan 0 \
    --batchSize 16 \
    --nz 200\
    --lambda_tv 1\
    --lambda_flow 0\
    --norm batch\
    --ignore_center\
    --train_split 0.8\
    --category human\


    #--nl_enc relu\



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




