import cv2
import os
import click
import numpy as np
import copy
import OpenEXR, Imath, Image
import random
import pandas as pd
import shutil
@click.command()

def pick():
    index_list = np.array([97560,18360,61560])
    view_list= np.arange(360)
    gt_folder = '/home/xu/data/view_synthesis/car_v1_test_random'
    pd_folder = '/home/xu/workspace/view_synthesis/results/car_exp/evaluate_normal'
    pd_folder_auto = '/home/xu/workspace/view_synthesis/results/car_exp_neighbour/evaluate_auto'

    output_folder='/home/xu/workspace/view_synthesis/exps/sampled_images/car'

    for i in index_list:
        path = os.path.join(output_folder, "%06d" % i)
        if not os.path.isdir(path):
            os.mkdir(path)

        gt_out_path = os.path.join(path, "gt")
        if not os.path.isdir(gt_out_path):
            os.mkdir(gt_out_path)

        pd_out_path = os.path.join(path, "pd")
        if not os.path.isdir(pd_out_path):
            os.mkdir(pd_out_path)

        pd_out_path_auto = os.path.join(path, "auto")
        if not os.path.isdir(pd_out_path_auto):
            os.mkdir(pd_out_path_auto)

    for v in view_list:
        gt_view_folder = os.path.join(gt_folder,"%d"%v)

        index_view_list = index_list + v
        for root, folder, files in os.walk(gt_view_folder):
            for file in sorted(files):
                for idx, idx_abs in enumerate(index_view_list):
                    if str(idx_abs) in file:
                        dst_path = os.path.join(output_folder, "%06d"%(index_list[idx]), 'gt')
                        shutil.copy(os.path.join(gt_view_folder,file),dst_path)

    index_view_list = index_list + 180
    pd_view_folder = os.path.join(pd_folder,"%02d"%0,'images')
    for root, folder, files in os.walk(pd_view_folder):
        for file in sorted(files):
            for idx, idx_abs in enumerate(index_view_list):
                if str(idx_abs) in file:

                    dst_path = os.path.join(output_folder, "%06d"%(index_list[idx]), 'pd')
                    shutil.copy(os.path.join(pd_view_folder,file),dst_path)

    index_view_list = index_list + 180
    pd_view_folder = os.path.join(pd_folder_auto,"%02d"%0,'images')
    for root, folder, files in os.walk(pd_view_folder):
        for file in sorted(files):
            for idx, idx_abs in enumerate(index_view_list):
                if str(idx_abs) in file:
                    dst_path = os.path.join(output_folder, "%06d"%(index_list[idx]), 'auto')
                    shutil.copy(os.path.join(pd_view_folder,file),dst_path)

if __name__ == '__main__':
    #segments2sketches()
    # simplify_images()
    pick()
