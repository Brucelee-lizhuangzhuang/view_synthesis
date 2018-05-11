#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/human/    \
	--name human_full_random \
	--dataset_mode aligned_multi_view_random \
	--model multi_view_depth_random\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 10\
    --no_flip \
    --nz 200\
    --norm batch\
    --train_split 0.8\
    --category human\
    --test_views 80

#    --n_bilinear_layers 3
#    --use_pyramid

#    --add_grid\

#    --which_direction BtoA\

