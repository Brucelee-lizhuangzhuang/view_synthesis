import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable,grad
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F


class Pix2PixSTLModel(BaseModel):
    def name(self):
        return 'Pix2PixSTLModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        opt.output_nc = opt.input_nc
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, 2, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, tanh=True)
        self.flow_remapper = networks.flow_remapper(size=opt.fineSize, batch=opt.batchSize,gpu_ids=opt.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        grid = np.zeros((opt.fineSize,opt.fineSize,2))

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid[i,j,0] = j
                grid[i,j,1] = i
        grid /= (opt.fineSize/2)
        grid -= 1
        self.grid = torch.from_numpy(grid).cuda().float() #Variable(torch.from_numpy(grid))
        self.grid = self.grid.view(1,self.grid.size(0),self.grid.size(1),self.grid.size(2))
        self.grid = Variable(self.grid)


        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        input_C = input['C']
        if len(self.gpu_ids) > 0:
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
        self.input_C = input_C

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

        self.forward_flow = self.netG(self.real_A)
        self.forward_flow = self.forward_flow.permute(0,2,3,1)
        # if self.opt.which_direction == "BtoA":
        #     self.forward_flow = -self.forward_flow

        self.forward_map  = torch.clamp(self.forward_flow + self.grid,-1.,1.)
        self.backward_map = self.flow_remapper(self.forward_map,self.real_C + self.grid)

        self.fake_B = F.grid_sample(self.real_A, self.backward_map)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B_flow = self.netG(self.real_A)
        self.fake_B_flow = self.fake_B_flow.permute(0,2,3,1)
        self.real_C = Variable(self.input_C)

        if self.opt.which_direction == "BtoA":
            self.fake_B_flow = -self.fake_B_flow

        self.sample_grid = torch.nn.functional.tanh(self.fake_B_flow + self.grid)
        self.fake_B = F.grid_sample(self.real_A, self.sample_grid )

        self.real_B = Variable(self.input_B)

        # self.fake_B = self.fake_B_flow
        # self.fake_B = self.fake_B.permute(0, 3, 1, 2)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.opt.lambda_gan * self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.opt.lambda_gan * self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.opt.lambda_gan * self.criterionGAN(pred_fake, True)

        # Second, G(A) = B

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_A
        self.loss_G_flow = self.criterionL1(self.forward_flow, self.real_C) * self.opt.lambda_flow
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_flow

        self.loss_G.backward(retain_graph=True)

        self.grad = grad(self.loss_G, self.forward_flow)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('G_flow', self.loss_G_flow.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        # fake_C = util.tensor2im(self.forward_flow.permute(0, 3, 1, 2).data)
        real_C = util.tensor2im(self.real_C.permute(0, 3, 1, 2).data)
        grad = util.tensor2im(self.grad[0].permute(0, 3, 1, 2).data)
        forward_map = util.tensor2im(self.forward_map.permute(0, 3, 1, 2).data)
        backward_map = util.tensor2im(self.backward_map.permute(0, 3, 1, 2).data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),\
                            ('real_C', real_C), ('grad', grad),\
                            ('forward_map', forward_map), ('backward_map', backward_map)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
