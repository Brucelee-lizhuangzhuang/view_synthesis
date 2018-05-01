import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch.nn.functional as F
import itertools
import torchvision

### Feature Transformer Network
### http://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
def rotation_tensor(yaw, n_comps, gpu):
    yaw = yaw.expand(n_comps,1,1)
    one = Variable(torch.ones(n_comps, 1, 1).cuda(gpu, async=True))
    zero = Variable(torch.zeros(n_comps, 1, 1).cuda(gpu, async=True))

    # print yaw, one, zero
    rot_z = torch.cat((
        torch.cat((yaw.cos(), -yaw.sin(), zero), 1),
        torch.cat((yaw.sin(), yaw.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return rot_z

class FTAE(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7, upsample='basic',
                 norm_layer=None, nl_layer=None, gpu_ids=[],nz=200):
        super(FTAE, self).__init__()
        self.gpu_ids = gpu_ids

        kw, padw = 4, 1
        enc = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            enc += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and n < n_layers-1:
                enc += [norm_layer(ndf * nf_mult)]
                enc += [nl_layer()]
        # sequence += [nn.AvgPool2d(8)]
        self.enc = nn.Sequential(*enc)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, nz*3), nn.LeakyReLU(0.2, True)])
        self.fc2 = nn.Sequential(*[nn.Linear(nz*3, ndf * nf_mult), nn.LeakyReLU(0.2, True)])

        deconv = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**(n_layers - n - 1), 4)
            deconv += networks.upsampleLayer(ndf * nf_mult_prev, ndf * nf_mult, upsample=upsample)
            if norm_layer is not None and (n_layers - n + 1) < n_layers:
                deconv += [norm_layer(ndf * nf_mult)]
            deconv += [nl_layer()]
        deconv += networks.upsampleLayer(ndf, 2, upsample='bilinear')
        deconv += [nn.Tanh()]

        self.deconv = nn.Sequential(*deconv)

        self.nz = nz

    def forward(self, x, yaw, Tz=0):
        z_conv = self.enc(x)
        z_fc = self.fc(z_conv.view(x.size(0),-1) ).view(x.size(0), self.nz, 3)
        z_fc = F.tanh(z_fc)

        R = rotation_tensor(yaw, x.size(0), self.gpu_ids[0])


        # R = np.array(
        #    [ [np.cos(yaw),-np.sin(yaw), 0],
        #     [np.sin(yaw), np.cos(yaw), 0],
        #     [        0,         0, 1]]
        # )
        # R = np.array(
        #    [[        1,         0, 0],
        #     [0,np.cos(yaw),-np.sin(yaw)],
        #     [0,np.sin(yaw), np.cos(yaw)]
        #     ]
        # )
        # R = np.array(
        #    [[np.cos(yaw),0,-np.sin(yaw)],
        #     [0, 1, 0],
        #     [np.sin(yaw),0, np.cos(yaw)]
        #     ]
        # )
        # R = np.eye(3,3)
        T = np.array([Tz,0,0])
        # R = Variable(torch.from_numpy(R.astype(np.float32))).cuda().expand(x.size(0),3,3)
        # T = Variable(torch.from_numpy(T.astype(np.float32))).cuda().expand(x.size(0),self.nz,3)
        z_rot = z_fc.bmm(R) # + T
        z_rot_fc = self.fc2(z_rot.view(x.size(0), self.nz*3))
        return self.deconv(z_rot_fc.view(z_conv.size(0),z_conv.size(1),z_conv.size(2),z_conv.size(3))), z_fc

class FTAEModel(BaseModel):
    def name(self):
        return 'FTAEModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.yaw = Variable(torch.Tensor([-np.pi/4.]).cuda(opt.gpu_ids[0], async=True), requires_grad=False)
        # load/define networks
        self.netG = FTAE(opt.input_nc,opt.ngf, n_layers=int(np.log2(opt.fineSize)), upsample=opt.upsample,
                         norm_layer = networks.get_norm_layer(norm_type=opt.norm),
                         nl_layer=networks.get_non_linearity(layer_type='lrelu'), gpu_ids=opt.gpu_ids, nz=opt.nz)
        if len(opt.gpu_ids) > 0:
            self.netG.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netG, init_type="normal")

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
            self.criterionTV = networks.TVLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),  #, [self.yaw]
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
        self.grid = self.grid.view(1,self.grid.size(0),self.grid.size(1),self.grid.size(2)).expand(opt.batchSize,opt.fineSize,opt.fineSize,2)
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


        # input_A = input['B']
        # input_B = flip(input_A,3)

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.dataset_mode == 'aligned_with_C':
            input_C = input['C']
            if len(self.gpu_ids) > 0:
                self.input_C = input_C.cuda(self.gpu_ids[0], async=True)

        #
        self.mask = torch.sum(self.input_B, dim=1)
        self.mask = (self.mask < 3.0).unsqueeze(1)
        self.mask = self.mask.expand(self.input_B.size(0),2,self.input_B.size(2),self.input_B.size(3))
        #
        self.mask0 = torch.sum(self.input_A, dim=1)
        self.mask0 = (self.mask0 < 3.0).unsqueeze(1)
        self.mask0 = self.mask0.expand(self.input_B.size(0),2,self.input_B.size(2),self.input_B.size(3))

    def forward(self):
        add_grid = self.opt.add_grid
        rectified = self.opt.rectified
        self.real_A = Variable(self.input_A)
        if self.opt.dataset_mode == 'aligned_with_C':
            self.real_C = Variable(self.input_C)+self.grid

        self.fake_B_flow,_ = self.netG(self.real_A, self.yaw)
        self.fake_B_flow_converted = convert_flow(self.fake_B_flow,self.grid,add_grid,rectified)
        self.fake_B = torch.nn.functional.grid_sample(self.real_A, self.fake_B_flow_converted)
        self.real_B = Variable(self.input_B)

        self.fake_B_0_flow,_  = self.netG(self.real_A, Variable(torch.Tensor([0        ]).cuda(self.gpu_ids[0], async=True)))
        self.fake_B_flow_converted0 = convert_flow(self.fake_B_0_flow,self.grid,add_grid,rectified)
        self.fake_B_0 = torch.nn.functional.grid_sample(self.real_A, self.fake_B_flow_converted0)

        self.fake_B_18_flow,_ = self.netG(self.real_A, Variable(torch.Tensor([-np.pi/8.]).cuda(self.gpu_ids[0], async=True)))
        self.fake_B_18 = torch.nn.functional.grid_sample(self.real_A, convert_flow(self.fake_B_18_flow,self.grid,add_grid,rectified))

    # no backprop gradients
    def test(self):
        add_grid = self.opt.add_grid
        rectified = self.opt.rectified
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B_list = []
        for i in range(10):
            fake_B_flow,z = self.netG(self.real_A, Variable(torch.Tensor([-i/9.*np.pi/4.]).cuda(self.gpu_ids[0], async=True)))
            fake_B = torch.nn.functional.grid_sample(self.real_A, convert_flow(fake_B_flow,self.grid,add_grid,rectified))
            self.fake_B_list.append(fake_B)
        # np.save(os.path.join("./results/features", os.path.basename(self.image_paths[0]) ), z.data.cpu().numpy())

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
        # Total variation loss

        self.loss_TV = self.criterionTV(self.fake_B_flow) * self.opt.lambda_tv
        self.loss_TV_2 = self.criterionTV(self.fake_B_0_flow) * self.opt.lambda_tv

        if self.opt.lambda_flow > 0:
            self.loss_G_flow = self.criterionL1(self.fake_B_flow_converted.permute(0,3,1,2)[self.mask],
                                                self.real_C.permute(0,3,1,2)[self.mask]) * self.opt.lambda_flow
        else:
            self.loss_G_flow = 0. * self.loss_TV

        if self.opt.lambda_flow0 > 0:
            self.loss_G_flow0 = self.criterionL1(self.fake_B_flow_converted.permute(0,3,1,2)[self.mask0],
                                                self.grid.permute(0,3,1,2)[self.mask0]) * self.opt.lambda_flow
        else:
            self.loss_G_flow0 = 0. * self.loss_TV


        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G_L1_2 = self.criterionL1(self.fake_B_0, self.real_A) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L1_2 \
                      + self.loss_TV + self.loss_TV_2 + self.loss_G_flow + self.loss_G_flow0

        self.loss_G.backward()

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
                            ('G_L1_2', self.loss_G_L1_2.data[0]),
                            ('F_L1', self.loss_G_flow.data[0]),
                            ('F_L10', self.loss_G_flow0.data[0]),
                            ('TV', self.loss_TV.data[0]),
                            ('TV2', self.loss_TV_2.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ('Yaw', self.yaw.data[0])
                            ])

    def get_current_visuals(self):
        if not self.opt.isTrain:
            return self.get_current_visuals_test()
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)

        fake_B_0 = util.tensor2im(self.fake_B_0.data)
        fake_B_18 = util.tensor2im(self.fake_B_18.data)

        flow = util.tensor2im(self.fake_B_flow_converted.permute(0,3,1,2).data)
        flow0 = util.tensor2im(self.fake_B_flow_converted0.permute(0,3,1,2).data)

        if self.opt.dataset_mode == 'aligned_with_C':
            real_flow = util.tensor2im(self.real_C.permute(0,3,1,2).data)
        else:
            real_flow = util.tensor2im(self.fake_B_flow_converted.permute(0, 3, 1, 2).data)


        return OrderedDict([('real_A', real_A), ('fake_B_36', fake_B), ('real_B', real_B),
                            ('fake_B_0', fake_B_0), ('fake_B_18', fake_B_18),
                            ('flow',flow), ('flow0',flow0), ('real_flow', real_flow)])

    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        visual_list = OrderedDict([('real_A', real_A)])
        for idx,fake_B_var in enumerate(self.fake_B_list):
            visual_list['%d'%idx] = util.tensor2im(fake_B_var.data)
        visual_list['real_B'] = real_B
        return visual_list

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

def convert_flow(flow, grid, add_grid=False, rectified=False):
    b,_,h,w = flow.size()
    flow_ret = flow.permute(0, 2, 3, 1)
    if rectified:

        flow_ret = torch.cat([flow_ret[:,:,:,0].unsqueeze(3),grid[:b,:,:,1].unsqueeze(3),], dim=3)
        if add_grid:
            grid_new = torch.cat([grid[:b,:,:,0].unsqueeze(3),Variable(torch.zeros(b,h,w,1).cuda()),], dim=3)
            flow_ret += grid_new
    elif add_grid:
        flow_ret = flow_ret + grid
    return flow_ret