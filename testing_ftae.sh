#!/bin/bash
dataset=$1
epoch=$2

python test.py \
	--dataroot ~/data/$dataset    \
	--name ${dataset}_ftae  \
	--dataset_mode aligned \
	--model ftae                     \
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 50\
    --no_flip \
    --serial_batch\
    --nz 200

#    --which_direction BtoA