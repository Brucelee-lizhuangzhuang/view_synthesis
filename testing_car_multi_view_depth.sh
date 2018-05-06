#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/car_multi_view/    \
	--name car_multi_view_depth_ftae_randomAB \
	--dataset_mode aligned_multi_view \
	--model multi_view_depth\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 60\
    --no_flip \
    --serial_batch\
    --nz 200\
    --norm batch\

#    --use_pyramid

#    --add_grid\

#    --which_direction BtoA\

