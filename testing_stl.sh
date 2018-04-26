#!/bin/bash
dataset=$1
epoch=$2

python test.py \
	--dataroot ~/data/$dataset    \
	--name ${dataset}_stl_flow_AtoB  \
	--dataset_mode aligned_with_flow \
	--model pix2pix_stl                     \
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --which_model_netG resnet_6blocks  \
    --which_model_netE resnet_128	   \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 50\
    --no_flip \
    --serial_batch\
    --which_direction AtoB

#    --which_direction BtoA