#!/bin/bash
dataset=$1
epoch=$2

python test.py \
	--dataroot ~/data/$dataset    \
	--name ${dataset}_pix2pix \
	--dataset_mode aligned \
	--model pix2pix                    \
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