#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/car/    \
	--name car64 \
	--dataset_mode aligned \
	--model ftae_flow\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 64	                   \
    --fineSize 64			   \
 	--how_many 60\
    --no_flip \
    --serial_batch\
    --nz 200\
    --norm batch\

#    --add_grid\

#    --which_direction BtoA\

