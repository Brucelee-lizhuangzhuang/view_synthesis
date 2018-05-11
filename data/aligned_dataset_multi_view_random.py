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
        self.nv = 18
        self.train_split = opt.train_split

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
            # index = np.random.randint(0, self.__len__())

        if self.opt.phase == 'test' :
            idx_A = np.random.randint(0, self.nv - 1) if self.opt.category == 'car'else int(self.nv/2)

            idx_B = idx_A
            idx_C = idx_A
            yaw1 = 0
            yaw2 = 0
        else:
            if self.opt.category in ['car','car1'] :
                idx_A = np.random.randint(0, self.nv - 1)
                choices = [2,1,-1,-2]

                if self.opt.ignore_center:
                    idx_B = np.random.randint(0, self.nv - 2)
                    idx_B = (self.nv-1) if idx_B == idx_A else idx_B
                else:
                    idx_B = np.random.randint(0, self.nv - 1)

                if self.opt.only_neighbour:
                    idx_B = idx_A + np.random.choice(choices)

                # if idx_B >= 1: choices.append(-1)
                # if idx_B <= self.nv-1 : choices.append(1)
                idx_C = (idx_B + np.random.choice(choices)) if self.opt.number_samples == 3 else idx_A

                yaw1 = -(idx_B-idx_A) * np.pi/9
                yaw2 = -(idx_B-idx_C) * np.pi/9

                idx_C = np.mod(idx_C,self.nv)
                idx_B = np.mod(idx_B,self.nv)


            if self.opt.category == 'human':
                idx_A = int(self.nv/2)
                idx_B = np.random.randint(0, self.nv - 1)
                # if idx_B >= 1: choices.append(-1)
                # if idx_B <= self.nv-1 : choices.append(1)
                idx_C = idx_A

                yaw1 = -(idx_B - idx_A) * np.pi / 18
                yaw2 = -(idx_B - idx_C) * np.pi / 18

        if self.opt.phase == 'test' :
            idx_C = idx_A

        if self.opt.category == 'car':
            bg_color = (255,255,255)
        if self.opt.category == 'car1':
            bg_color = (64,64,64)
        if self.opt.category == 'human':
            bg_color = (0,0,0)

        A = Image.open(self.paths[idx_A][index]).convert('RGB')
        A,_ = self.remapping_background(A, bg_color)
        A = self.transform(A)

        B = Image.open(self.paths[idx_B][index]).convert('RGB')
        B,mask = self.remapping_background(B, bg_color)
        mask = np.logical_not(mask)
        mask = mask.astype(np.float32)
        mask[mask==0] = -2
        B = self.transform(B)

        C = Image.open(self.paths[idx_C][index]).convert('RGB')
        C,_ = self.remapping_background(C, bg_color)
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


        return {'A': A, 'B': B, 'C': C, 'YawAB': torch.Tensor([yaw1]),'YawCB': torch.Tensor([yaw2]),'mask': torch.Tensor( mask),
                'A_paths': self.paths[int(self.nv/2)][index], }

    def __len__(self):
        if self.opt.phase == 'train':
            return int(len(self.paths[int(self.nv/2)])*self.train_split)
        else:
            return int(len(self.paths[int(self.nv/2)])*(1-self.train_split) )

    def name(self):
        return 'AlignedDatasetMultiView'

    def remapping_background(self, image, bg_color):
        data = np.array(image)

        r1, g1, b1 = bg_color  # Original value
        r2, g2, b2 = 128, 128, 128  # Value that we want to replace it with

        red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        data[:, :, :3][mask] = [r2, g2, b2]

        return Image.fromarray(data),mask

