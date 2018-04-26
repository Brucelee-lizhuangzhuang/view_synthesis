import os.path
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_label
from PIL import Image
import PIL
import random
import numpy

class UnalignedTensorDatasetWithLabel(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        # New: make dataset

        self.A_paths = make_dataset_label(self.dir_A)
        self.B_paths = make_dataset_label(self.dir_B)
        # New: make dataset
        if self.opt.cond_nc > 0:
            self.C_paths = make_dataset_label(self.dir_C)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if self.opt.cond_nc > 0:
            self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
            self.C_paths = sorted(self.C_paths)
            self.C_size = len(self.C_paths)

            if self.B_size != self.C_size:
                raise ValueError("B and C should be paired and have the same length!")

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_arr = numpy.load(A_path)
        B_arr = numpy.load(B_path)
        # add noise
        # A_arr = A_arr + (0.25 * numpy.random.randn(A_arr.shape[0],A_arr.shape[1],A_arr.shape[2]) + 0.25)
        # B_arr = B_arr + (0.25 * numpy.random.randn(B_arr.shape[0],B_arr.shape[1],B_arr.shape[2]) + 0.25)
        A_labels = numpy.argmax(A_arr, axis=0)
        B_labels = numpy.argmax(B_arr, axis=0)

        A = torch.from_numpy(A_arr).float()
        B = torch.from_numpy(B_arr).float()
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if self.opt.cond_nc > 0:
            C_path = self.C_paths[index_B]
            C_arr = numpy.loadtxt(C_path, dtype=numpy.float32)
            C_arr = numpy.reshape(C_arr, (C_arr.shape[0], 1, 1))
            C_arr = numpy.repeat(C_arr, self.opt.fineSize, axis=1)
            C_arr = numpy.repeat(C_arr, self.opt.fineSize, axis=2)
            C = torch.from_numpy(C_arr)
            return {'A': A, 'B': B, 'C': C, 'A_labels': A_labels, 'B_labels': B_labels,
                    'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}
        else:
            return {'A': A, 'B': B, 'A_labels': A_labels, 'B_labels': B_labels,
                    'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedTensorDatasetWithLabel'
