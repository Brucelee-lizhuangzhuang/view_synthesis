#!/bin/bash
epoch=$1

python test.py \
	--dataroot ~/data/view_synthesis/pvhm_test/    \
	--name ftae_flow_rect_grid_tv_bilinear \
	--dataset_mode aligned_with_C \
	--model ftae_flow\
    --phase test        		   \
	--no_dropout                       \
	--which_epoch $epoch               \
    --loadSize 128	                   \
    --fineSize 128			   \
 	--how_many 15\
    --no_flip \
    --serial_batch\
    --nz 200\
    --rectified\
    --add_grid\
    --which_direction BtoA\

#    --which_direction BtoA