#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/car_full/    \
	--name car_full_random_neighbour \
	--dataset_mode aligned_multi_view_random \
	--model multi_view_depth_random\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 256	                   \
    --fineSize 256			   \
 	--how_many 10\
    --no_flip \
    --serial_batch\
    --nz 200\
    --norm batch\
    --train_split 0.8\
    --test_views 160

#    --n_bilinear_layers 3
#    --use_pyramid

#    --add_grid\

#    --which_direction BtoA\

