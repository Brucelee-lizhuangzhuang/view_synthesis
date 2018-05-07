import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset,make_dataset_label
from PIL import Image
import numpy as np

class AlignedDatasetMultiView(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dirs = []
        self.paths = []
        self.random_AB = opt.random_AB
        self.nv = 9
        self.train_split = .8

        for i in range(self.nv):
            self.dirs.append(os.path.join(opt.dataroot, "%d" %i) )
            self.paths.append(sorted(make_dataset(self.dirs[i]) ) )

        # if opt.phase == 'test':
        #     self.dirs[self.center_view] = os.path.join(opt.dataroot, "test")
        #     self.paths.append(sorted(make_dataset(self.dirs[self.center_view])))

        # self.dir_C  = os.path.join(opt.dataroot, opt.phase+"C")
        # self.C_paths = sorted(make_dataset_label(self.dir_C))

        self.transform = get_transform(opt)

    def __getitem__(self, index):


        if self.opt.phase == 'test':
            index += int(len(self.paths[int(self.nv/2)])*self.train_split)+1

        idx_A = np.random.randint(0, self.nv - 1)
        idx_B = np.random.randint(0, self.nv - 1)
        choices = []
        if idx_B >= 1: choices.append(-1)
        if idx_B <= self.nv-1 : choices.append(1)
        idx_C = idx_B + np.random.choice(choices)

        if self.opt.phase == 'test':
            idx_A = 2
            idx_B = 2
            idx_C = 2




        A = Image.open(self.paths[idx_A][index]).convert('RGB')
        A = self.transform(A)
        B = Image.open(self.paths[idx_B][index]).convert('RGB')
        B = self.transform(B)
        C = Image.open(self.paths[idx_C][index]).convert('RGB')
        C = self.transform(C)


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        yaw1 = -(idx_B-idx_A) * np.pi/9
        yaw2 = -(idx_B-idx_C) * np.pi/9

        return {'A': A, 'B': B, 'C': C, 'YawAB': torch.Tensor([yaw1]),'YawCB': torch.Tensor([yaw2]),
                'A_paths': self.paths[int(self.nv/2)][index], }

    def __len__(self):
        if self.opt.phase == 'train':
            return int(len(self.paths[int(self.nv/2)])*self.train_split)
        else:
            return int(len(self.paths[int(self.nv/2)])*(1-self.train_split) )

    def name(self):
        return 'AlignedDatasetMultiView'
