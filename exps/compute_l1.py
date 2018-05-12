import cv2
import os
import click
import numpy as np
import copy
import matplotlib.pyplot as plt
@click.command()
@click.option("--pd_folder", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/view_synthesis/results/car_exp/test_latest/images', help="Input image.")
@click.option("--gt_folder", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/surreal/car_shapenet1_full_view_test/rgb', help="Output dest")
def convert(pd_folder, gt_folder):
    gt_list = []
    pd_list = []

    N = 3600
    for root, folder, files in os.walk(gt_folder):
        for i,file in enumerate(sorted(files)):
            gt_list.append(os.path.join(gt_folder,file))
            if i >= N: break
            print i

    for root, folder, files in os.walk(pd_folder):
        for i,file in enumerate(sorted(files)):
            pd_list.append(os.path.join(pd_folder, file))
            if i >= N: break

    errors = np.zeros(360)
    errors2 = np.zeros(360)
    for i,(gt_path,pd_path) in enumerate(zip(gt_list,pd_list)):


        img_gt = cv2.imread(gt_path)
        img_pd = cv2.imread(pd_path)
        if np.mod(i,10) == 0:
            img_bl = img_gt.copy()
        length = 256*256


        mask = np.where((img_gt==(64,64,64)).all(axis=2))

        loss = compute_loss(img_gt,img_pd,mask)
        loss2 = compute_loss(img_gt,img_bl,mask)


        # cv2.imshow('s',diff.astype(np.uint8))
        # cv2.waitKey()
        # diff2 = diff[mask].flatten()
        # print diff2.shape
        print i
        errors[ np.mod(i,360)] += loss
        errors2[ np.mod(i,360)] += loss2

    plt.plot(np.arange(360), errors,np.arange(360), errors2)

    plt.show()

def compute_loss(img1, img2, mask):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    diff = np.abs(diff)
    diff[mask] = 0
    diff = np.sum(diff) / (255*255 - mask[0].shape[0]) / 3. / 255.
    return diff


if __name__ == '__main__':
    #segments2sketches()
    # simplify_images()
    convert()
