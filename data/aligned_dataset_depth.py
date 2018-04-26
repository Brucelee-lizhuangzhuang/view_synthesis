import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset,make_dataset_label
from PIL import Image
import numpy as np

class AlignedDatasetDepth(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase+"A")
        self.dir_B  = os.path.join(opt.dataroot, opt.phase+"B")
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset_label(self.dir_B))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)

        B_path = self.B_paths[index]
        B_arr = np.load(B_path)
        B = torch.from_numpy(B_arr).float().view(1,128,128).expand(3,128,128)

        # w = A.size(2)
        # h = A.size(1)
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w + w_offset:w + w_offset + self.opt.fineSize]

        # B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc
        #
        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)
        # if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)
        #
        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetDepth'
