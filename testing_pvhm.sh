#!/bin/bash
type=$1
epoch=$2

python test.py \
	--dataroot ~/data/view_synthesis/pvhm_test/    \
	--name ${type}_refine  \
	--dataset_mode aligned_with_C \
	--model ${type}_refine                     \
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --which_model_netG unet_128  \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 50\
    --no_flip \
    --serial_batch\
    --which_direction BtoA

#    --which_direction BtoA