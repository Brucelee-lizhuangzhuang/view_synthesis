#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/car_multi_view_elevated//    \
	--name car_multi_view_depth_ftae_elevated \
	--dataset_mode aligned_multi_view \
	--model multi_view_depth\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 10\
    --no_flip \
    --serial_batch\
    --nz 200\
    --norm batch\
    --train_split 0.8

#    --n_bilinear_layers 3
#    --use_pyramid

#    --add_grid\

#    --which_direction BtoA\

