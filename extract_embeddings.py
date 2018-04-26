import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import save_images_vae
from itertools import islice
from util import html
from util import util
import numpy as np

# helper function
# options
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
print('Loading model %s' % opt.model)

embeddings = []
# test stage
for i, data in enumerate(islice(dataset, opt.how_many)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))
    _, _, _, _, z = model.test_simple(None, encode_real_B=True)
    z = z.cpu().data.numpy()
    embeddings.append(z.copy())

np.save("/home/xu/embeddings.npy",np.array(embeddings))
