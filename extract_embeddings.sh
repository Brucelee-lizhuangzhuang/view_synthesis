#!/bin/bash
dataset_name=$1
epoch=$2
model=vae_cycle_gan_all

python extract_embeddings.py \
	--dataroot ~/data/$dataset_name    \
	--name ${dataset_name}_vae_cyclegan_all_2layers \
	--model $model                     \
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --which_model_netG resnet_6blocks  \
    --which_model_netE resnet_128	   \
    --loadSize 143	                   \
    --fineSize 128			   \
    --resize_or_crop test\
    --how_many 1000\
    --nz 16\
    --serial_batch\

#    --random_walk