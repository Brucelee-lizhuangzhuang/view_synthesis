import cv2
import os
import click
import numpy as np
import copy
import matplotlib.pyplot as plt
@click.command()
# @click.option("--record_folder", type=click.Path(exists=True, dir_okay=True),
#               default='/home/xu/workspace/view_synthesis/exps/records/', help="Input image.")

def sample():
    N = 100000
    idxA = np.random.randint(0,360,(N,1))
    idxM = np.random.randint(0,400,(N,1))
    deltas_range = np.arange(-40,40)
    deltas_choices = np.random.choice(deltas_range,(N,1))
    idxB = (idxA + deltas_choices) #% 360

    list = np.concatenate((idxA, idxB, idxM),axis=1)
    np.savetxt('sample_list.txt',list.astype(np.int),fmt='%d')
    # plt.hist(idxA)
    # plt.show()




if __name__ == '__main__':
    #segments2sketches()
    # simplify_images()
    sample()
