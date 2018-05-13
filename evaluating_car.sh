#!/bin/bash
epoch=$1

python evaluate.py \
	--dataroot ~/data/view_synthesis/car_v1_test/    \
	--name car_exp_neighbour \
	--dataset_mode aligned_multi_view_random \
	--model multi_view_depth_random\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 256	                   \
    --fineSize 256			   \
 	--how_many 400\
    --no_flip \
    --serial_batch\
    --nz 200\
    --norm batch\
    --train_split 0\
    --test_views 80 \
    --category car1\
    --only_neighbour\
    --auto_aggressive


    #--auto_aggressive

#    --n_bilinear_layers 3
#    --use_pyramid

#    --add_grid\

#    --which_direction BtoA\

