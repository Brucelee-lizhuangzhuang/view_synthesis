import cv2
import os
import click
import numpy as np
import copy
import matplotlib.pyplot as plt
@click.command()
@click.option("--record_folder", type=click.Path(exists=True, dir_okay=True),
              default='/home/xu/workspace/view_synthesis/exps/records/', help="Input image.")

def plot(record_folder):


    for root, folders, files in os.walk(record_folder):
        for i,file in enumerate(sorted(files)):
            errors = np.load(os.path.join(record_folder,file))

            plt.plot(np.arange(360),errors)
            plt.show()

if __name__ == '__main__':
    #segments2sketches()
    # simplify_images()
    plot()
