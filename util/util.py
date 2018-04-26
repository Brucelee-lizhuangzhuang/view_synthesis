from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections

cm_cloth = np.load('./util/cm_cloth.npy')
cm_body = np.load('./util/cm_body.npy')


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 2:
        image_numpy = image_numpy.reshape((1,image_numpy.shape[0],image_numpy.shape[1]))
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def class_tensor2im(class_tensor, colormap='cm_cloth', imtype=np.uint8):
    if colormap == 'cm_cloth':
        cm = cm_cloth
    elif colormap == 'cm_body':
        cm = cm_body
    else:
        raise ValueError("Unknown colormap.")
    class_numpy = class_tensor[0].cpu().float().numpy()
    labels = np.argmax(class_numpy,axis=0)
    images = np.zeros( (labels.shape[0], labels.shape[1], 3) )
    for i in range(len(cm)):
        images[(np.where((labels == i)))] = cm[i]
    images = images[:,:,::-1]

    return images.astype(imtype)

def class_tensor2im_A(class_tensor, imtype=np.uint8):
    return class_tensor2im(class_tensor, colormap='cm_body',imtype=imtype)

def class_tensor2im_B(class_tensor, imtype=np.uint8):
    return class_tensor2im(class_tensor, colormap='cm_cloth', imtype=imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
