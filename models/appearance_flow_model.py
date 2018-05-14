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
from projection_layer import inverse_warp
import projection_layer2

### Feature Transformer Network
### http://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class AFN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7, n_bilinear_layers=0,
                 norm_layer=None, nl_layer_enc=None, nl_layer_dec=None, gpu_ids=[], nz=200, use_vae=False, pred_mask=False):
        super(AFN, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_vae = use_vae
        self.pred_mask = pred_mask
        kw, padw = 4, 1

        enc = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer_enc()]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            enc += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and n < n_layers - 1:
                enc += [norm_layer(ndf * nf_mult)]
            enc += [nl_layer_enc()]
        self.enc = nn.Sequential(*enc)

        self.fc1 = nn.Sequential(*[nn.Linear(4*ndf * nf_mult, 4096),nn.BatchNorm1d(4096),nl_layer_enc(),
                    nn.Linear(4096, 4096),nn.BatchNorm1d(4096), nl_layer_enc()])

        self.fc2 = nn.Sequential(*[nn.Linear(4096+256, 4096),nn.BatchNorm1d(4096), nl_layer_enc(),
                  nn.Linear(4096, 4*ndf*nf_mult), nn.BatchNorm1d(4*ndf*nf_mult),nl_layer_enc()])
        deconv = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 8)

            upsample = 'bilinear' if n_layers - n < n_bilinear_layers else 'basic'
            deconv += networks.upsampleLayer(ndf * nf_mult_prev, ndf * nf_mult, upsample=upsample)
            if norm_layer is not None and (n_layers - n + 1) < n_layers:
                deconv += [norm_layer(ndf * nf_mult)]
            deconv += [nl_layer_dec()]

        if n_bilinear_layers > 0:
            deconv += networks.upsampleLayer(ndf, 2, upsample='bilinear')
        else:
            deconv += networks.upsampleLayer(ndf, 2, upsample='basic')
        deconv += [nn.Tanh()]
        self.deconv = nn.Sequential(*deconv)

        self.view_decoder = nn.Sequential(*[nn.Linear(nz, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2, True),
                        nn.Linear(128, 256), nn.BatchNorm1d(256),nn.LeakyReLU(0.2, True)])


    def forward(self, x, t):
        z = self.enc(x)
        b,c,h,w = z.size()
        z = self.fc1(z.view(b,-1))
        z = torch.cat([z, self.view_decoder(t)],dim=1)
        z = self.fc2(z).view(b,c,h,w)
        return self.deconv(z)


class AppearanceFlowModel(BaseModel):
    def name(self):
        return 'AppearanceFlowModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # self.yaw = Variable(torch.Tensor([np.pi/4.]).cuda(opt.gpu_ids[0], async=True), requires_grad=False)
        # load/define networks
        input_nc = opt.input_nc + 2 if opt.concat_grid else opt.input_nc

        if self.opt.view_representation == 'index':
            nz = 18
        elif self.opt.view_representation == 'cos_sin':
            nz = 2
        else:
            raise NotImplementedError('only support cos-sin or index')

        self.netG = AFN(input_nc, opt.ngf, n_layers=int(np.log2(opt.fineSize))-1, n_bilinear_layers=opt.n_bilinear_layers,
                         norm_layer=networks.get_norm_layer(norm_type=opt.norm),
                         nl_layer_enc=networks.get_non_linearity(layer_type=opt.nl_enc),
                         nl_layer_dec=networks.get_non_linearity(layer_type=opt.nl_dec),gpu_ids=opt.gpu_ids,
                         nz=nz, use_vae=opt.use_vae, pred_mask=opt.pred_mask)

        if len(opt.gpu_ids) > 0:
            self.netG.cuda(opt.gpu_ids[0])
        networks.init_weights(self.netG, init_type="normal")

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),  #, [self.yaw]
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
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

        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_mask = input['mask']
        input_T = input['T']


        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_T = input_T.cuda(self.gpu_ids[0], async=True)

            input_mask = input_mask.cuda(self.gpu_ids[0], async=True)


        self.input_A = input_A
        self.input_B = input_B
        self.input_mask = input_mask
        self.input_T = input_T

        self.image_paths = input['A_paths' if AtoB else 'B_paths']


        self.maskB_fg = torch.sum(self.input_B, dim=1)
        self.maskB_fg = (self.maskB_fg < 3.0).unsqueeze(1)
        self.maskB_fg = self.maskB_fg.expand(self.input_B.size(0),3,self.input_B.size(2),self.input_B.size(3))
        #
        self.maskA_fg = torch.sum(self.input_A, dim=1)
        self.maskA_fg = (self.maskA_fg < 3.0).unsqueeze(1)
        self.maskA_fg = self.maskA_fg.expand(self.input_A.size(0),3,self.input_A.size(2),self.input_A.size(3))

        if self.opt.category == 'car1':
            return

#        self.input_A[self.maskA] = 0.
#        self.input_B[self.maskB] = 0.

    def forward(self):

        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_mask = Variable(self.input_mask)

        b,c,h,w = self.real_A.size()

        if self.opt.view_representation == 'index':
            view_indexes = self.input_T.cpu().numpy().astype(np.int)
            view_code = np.zeros((b,18))
            for i,j in enumerate(view_indexes):
                view_code[i,j] = 1
            self.real_T = Variable( torch.from_numpy(view_code.astype(np.float32)).cuda()).view(b,18)
        elif self.opt.view_representation == 'cos_sin':
            angle = Variable(self.input_T) * np.pi/9
            self.real_T = torch.cat([angle.cos(), angle.sin()],dim=1)
        else:
            raise NotImplementedError('only support cos-sin or index')


        self.flow = self.netG(self.real_A, self.real_T).permute(0,2,3,1) + self.grid[:b,:,:,:]
        self.fake_B = F.grid_sample(self.real_A, self.flow)
        #
        # if self.opt.use_masked_L1:
        #     self.fake_B = self.fake_B*self.real_mask.unsqueeze(1).expand(b,3,h,w)


    # no backprop gradients
    def test(self):

        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.real_B = Variable(self.input_B, volatile=True)
            self.fake_B_list = []

            NV = self.opt.test_views
            b,c,h,w = self.real_A.size()


            yaw = 0
            real_A = self.real_A
            for i in range(NV):
                yaw += 2*np.pi/NV
                self.fake_B_list.append(self.fake_B)

                if self.opt.auto_aggressive and np.mod(i, NV/9) == 0:
                    real_A = self.fake_B
                    yaw =0


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        self.loss_TV = self.criterionTV(self.flow) * self.opt.lambda_tv

        # Second, G(A) = B
        self.loss_G_L1_masked = self.criterionL1(self.fake_B[self.maskB_fg], self.real_B[self.maskB_fg]) * self.opt.lambda_A
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_L1 + self.loss_TV #  #+ self.loss_kl #+ self.loss_G_L1_masked

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_L1', self.loss_G_L1.data[0]),
                            ('G_L1_masked', self.loss_G_L1_masked.data[0]),
                            ('TV', self.loss_TV.data[0]),
                            ])

    def get_current_visuals(self):
        if not self.opt.isTrain:
            return self.get_current_visuals_test()
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
#        real_C = util.tensor2im(self.real_C.data)

        flow = util.tensor2im(self.flow.permute(0,3,1,2).data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                            ('flow',flow)])

    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        visual_list = OrderedDict([])
        for idx,fake_B_var in enumerate(self.fake_B_list):
            visual_list['%03d'%idx] = util.tensor2im(fake_B_var.data)
        # visual_list['real_B'] = real_B
        return visual_list

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

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
        flow_ret = flow_ret + grid[:b,:,:,:]
    return flow_ret

def get_z_random(batchSize, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batchSize, nz) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batchSize, nz)
    z = Variable(z.cuda())
    return z
