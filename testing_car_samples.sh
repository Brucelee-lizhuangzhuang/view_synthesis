#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/car_v1_test_random/    \
	--name car_exp \
	--dataset_mode aligned_multi_view_random \
	--model multi_view_depth_random\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 256	                   \
    --fineSize 256			   \
 	--how_many 100000\
    --no_flip \
    --serial_batch\
    --nz 200\
    --norm batch\
    --train_split 0\
    --test_views 80\
    --category car1\
    --list_path /home/xu/workspace/view_synthesis/exps/sample_list.txt


    #--auto_aggressive

#    --n_bilinear_layers 3
#    --use_pyramid

#    --add_grid\

#    --which_direction BtoA\

