import cv2
import os
import click
import numpy as np
import copy
import matplotlib.pyplot as plt
@click.command()
@click.option("--error_path", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/view_synthesis/results/car_exp/test_200/errors.txt', help="Input image.")
def plot(error_path):
    sample_list = np.loadtxt('/home/xu/workspace/view_synthesis/exps/sample_list.txt', dtype=int)
    errors_list = np.loadtxt(error_path)
    sample_list = sample_list[:errors_list.shape[0],:]


    angle_diff_list = sample_list[:,1] - sample_list[:,0]
    occs = np.zeros(80)
    errors = np.zeros(80)

    for diff,error in zip(angle_diff_list,errors_list):
        diff += 40
        occs[diff] += 1
        errors[diff] += error
    errors = errors / occs

    plt.plot(np.arange(80),errors)
    plt.show()
    print sample_list.shape, errors_list.shape

if __name__ == '__main__':

    plot()
