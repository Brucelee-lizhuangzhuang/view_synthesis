import cv2
import os
import click
import numpy as np
import copy
import matplotlib.pyplot as plt
@click.command()
@click.option("--pd_folder", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/view_synthesis/results/car_exp_neighbour/evaluate/', help="Input image.")
@click.option("--gt_folder", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/surreal/car_shapenet1_full_view_test/rgb', help="Output dest")
def convert(pd_folder, gt_folder):
    gt_list = []
    pd_list = []

    N = 3600
    for root, folders, files in os.walk(gt_folder):
        for i,file in enumerate(sorted(files)):
            if i%360 == 0:
                gt_list.append(list([]))
            gt_list[int(i/360)].append(os.path.join(gt_folder,file))
            print i

    for root, folders, _ in os.walk(pd_folder):
        for i,folder in enumerate(sorted(folders)):
            folder_full = os.path.join(root,folder,'images')
            if not os.path.exists(folder_full):
                continue
            pd_list.append(list([]))
            print folder_full
            for _, _, files in os.walk(folder_full):
                for j,file in enumerate(sorted(files)):
                    pd_list[i].append(os.path.join(folder_full, file))



    view_gap = 360/len(pd_list)
    for i,views in enumerate(pd_list):
        print i
        angles = (np.arange(-40, 40) + i*view_gap ) % 360
        errors = np.zeros(360)
        errors2 = np.zeros(360)
        for j,pd_path in enumerate(pd_list[i]):
            if j > 3200:
                break
            idx_model = j/80
            idx_angle = j%80
            a = angles[idx_angle]
            gt_path = gt_list[idx_model][a]

            img_gt = cv2.imread(gt_path)
            img_pd = cv2.imread(pd_path)

            mask = np.where((img_gt==(64,64,64)).all(axis=2))

            loss,diff = compute_loss(img_gt,img_pd,mask)
            print loss
            if  j%80 > 1:
                loss2,diff2 = compute_loss(img_gt,img_gt_prev_prev,mask)
                errors2[ a] += loss2
                # cv2.imshow('s2', diff2.astype(np.uint8))

            if j%80 > 0:
                img_gt_prev_prev = copy.copy(img_gt_prev)
            img_gt_prev = copy.copy(img_gt)

            # cv2.imshow('s',diff.astype(np.uint8))
            # cv2.waitKey()
            print j
            errors[ a] += loss

        plt.plot(np.arange(360), errors,np.arange(360), errors2)
        plt.show()

def compute_loss(img1, img2, mask):
    diff1 = img1.astype(np.float32) - img2.astype(np.float32)
    diff = np.abs(diff1)
    diff[mask] = 0
    loss = np.sum(diff) / (255*255 - mask[0].shape[0]) / 3. / 255.
    return loss, diff


if __name__ == '__main__':
    #segments2sketches()
    # simplify_images()
    convert()
