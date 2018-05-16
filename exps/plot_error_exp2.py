import cv2
import os
import click
import numpy as np
import copy
import matplotlib.pyplot as plt
@click.command()
@click.option("--error_path", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/view_synthesis/results/car_density_9views/test_200/errors.txt', help="Input image.")
def plot(error_path):
    N = 50000
    sample_list = np.loadtxt('/home/xu/workspace/view_synthesis/exps/sample_list.txt', dtype=int)[:N,:]
    errors_list_18views = np.loadtxt('/home/xu/workspace/view_synthesis/results/car_exp_neighbour/test_200/errors.txt')[:N]
    errors_list_9views = np.loadtxt('/home/xu/workspace/view_synthesis/results/car_density_9views/test_200/errors.txt')[:N]
    errors_list_6views = np.loadtxt('/home/xu/workspace/view_synthesis/results/car_density_6views/test_200/errors.txt')[:N]
    errors_list_flow = np.loadtxt('/home/xu/workspace/view_synthesis/results/car_flow/test_200/errors.txt')[:N]

    angle_diff_list = sample_list[:,1] - sample_list[:,0]
    errors_18views = calc_errors(errors_list_18views,angle_diff_list)
    errors_9views = calc_errors(errors_list_9views,angle_diff_list)
    errors_6views = calc_errors(errors_list_6views,angle_diff_list)
    errors_flow = calc_errors(errors_list_flow,angle_diff_list)

    plt.plot(np.arange(80),errors_18views, label='18')
    plt.plot(np.arange(80),errors_9views,label='9')
    plt.plot(np.arange(80),errors_6views,label='6')
    plt.plot(np.arange(80),errors_flow,label='flow')

    plt.legend()
    plt.show()


def calc_errors(errors_list, angle_diff_list):
    occs = np.zeros(80)
    errors = np.zeros(80)

    for diff, error in zip(angle_diff_list, errors_list):
        diff += 40
        occs[diff] += 1
        errors[diff] += error
    errors = errors / occs
    return errors

if __name__ == '__main__':

    plot()
