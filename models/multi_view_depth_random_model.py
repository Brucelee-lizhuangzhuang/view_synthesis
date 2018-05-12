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
def rotation_z_tensor(yaw, n_comps, gpu):
    yaw = yaw.unsqueeze(1)
    one = Variable(torch.ones(n_comps, 1, 1).cuda(gpu, async=True))
    zero = Variable(torch.zeros(n_comps, 1, 1).cuda(gpu, async=True))

    # print yaw, one, zero
    rot_z = torch.cat((
        torch.cat((yaw.cos(), -yaw.sin(), zero), 1),
        torch.cat((yaw.sin(), yaw.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return rot_z

def rotation_tensor(theta, phi, psi):
    n_comps = theta.size(0)
    theta = theta.unsqueeze(1)
    phi = phi.unsqueeze(1)
    psi = psi.unsqueeze(1)

    one = Variable(torch.ones(n_comps, 1, 1)).cuda()
    zero = Variable(torch.zeros(n_comps, 1, 1)).cuda()
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))

def rotation_x_tensor(yaw, n_comps, gpu):
    yaw = yaw.unsqueeze(1)
    one = Variable(torch.ones(n_comps, 1, 1).cuda(gpu, async=True))
    zero = Variable(torch.zeros(n_comps, 1, 1).cuda(gpu, async=True))

    # print yaw, one, zero
    rot_z = torch.cat((
        torch.cat((yaw.cos(), zero, -yaw.sin()), 1),
        torch.cat((yaw.sin(), yaw.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return rot_z

class FTAE(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7, n_bilinear_layers=0,
                 norm_layer=None, nl_layer_enc=None, nl_layer_dec=None, gpu_ids=[], nz=200, use_vae=False, pred_mask=False):
        super(FTAE, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_vae = use_vae
        self.pred_mask = pred_mask
        kw, padw = 4, 1

        enc = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer_enc()]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            enc += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and n < n_layers - 1:
                enc += [norm_layer(ndf * nf_mult)]
                enc += [nl_layer_enc()]
        # sequence += [nn.AvgPool2d(8)]
        self.enc = nn.Sequential(*enc)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, nz * 3)]) #, nn.LeakyReLU(0.2, True)
        if use_vae:
            self.fc_var = nn.Sequential(*[nn.Linear(ndf * nf_mult, nz * 3), nn.LeakyReLU(0.2, True)])
        self.fc2 = nn.Sequential(*[nn.Linear(nz * 3, ndf * nf_mult), nn.LeakyReLU(0.2, True)])

        deconv = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 4)

            upsample = 'bilinear' if n_layers - n < n_bilinear_layers else 'basic'
            deconv += networks.upsampleLayer(ndf * nf_mult_prev, ndf * nf_mult, upsample=upsample)
            if norm_layer is not None and (n_layers - n + 1) < n_layers:
                deconv += [norm_layer(ndf * nf_mult)]
            deconv += [nl_layer_dec()]

        output_nc = 2 if pred_mask else 1
        if n_bilinear_layers > 0:
            deconv += networks.upsampleLayer(ndf, output_nc, upsample='bilinear')
        else:
            deconv += networks.upsampleLayer(ndf, output_nc, upsample='basic')

        # deconv += [nn.Tanh()]

        self.deconv = nn.Sequential(*deconv)
        self.nz = nz

    def forward(self, x, R, Tz=0 ):
        z_conv = self.enc(x)
        if self.use_vae:
            mu = self.fc(z_conv.view(x.size(0),-1) )
            logvar = self.fc_var(z_conv.view(x.size(0),-1) )
            std = logvar.mul(0.5).exp_()
            eps = get_z_random(std.size(0), std.size(1), 'gauss')
            z_fc = eps.mul(std).add_(mu).view(x.size(0), self.nz, 3)
            T = np.array([1, 0, 0])
            T = Variable(torch.from_numpy(T.astype(np.float32))).cuda().expand(x.size(0), self.nz, 3)
            z_fc += T
            self.mu = mu
            self.logvar = logvar
        else:
            z_fc = self.fc(z_conv.view(x.size(0),-1) ).view(x.size(0), self.nz, 3)
        z_fc = F.tanh(z_fc)

        z_rot = z_fc.bmm(R) # + T
        z_rot_fc = self.fc2(z_rot.view(x.size(0), self.nz*3))

        output = self.deconv(z_rot_fc.view(z_conv.size(0),z_conv.size(1),z_conv.size(2),z_conv.size(3)))

        return output


    def get_mu_var(self):
        return self.mu,self.logvar


class MultiViewDepthModel(BaseModel):
    def name(self):
        return 'MultiViewDepthModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # self.yaw = Variable(torch.Tensor([np.pi/4.]).cuda(opt.gpu_ids[0], async=True), requires_grad=False)
        # load/define networks
        input_nc = opt.input_nc + 2 if opt.concat_grid else opt.input_nc

        self.netG = FTAE(input_nc, opt.ngf, n_layers=int(np.log2(opt.fineSize)), n_bilinear_layers=opt.n_bilinear_layers,
                         norm_layer=networks.get_norm_layer(norm_type=opt.norm),
                         nl_layer_enc=networks.get_non_linearity(layer_type=opt.nl_enc),
                         nl_layer_dec=networks.get_non_linearity(layer_type=opt.nl_dec),gpu_ids=opt.gpu_ids,
                         nz=opt.nz, use_vae=opt.use_vae, pred_mask=opt.pred_mask)

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

        zeros = Variable(torch.zeros((opt.batchSize,1)).cuda())
        ones = Variable(torch.ones((opt.batchSize,1)).cuda())
        if self.opt.category == 'car':
            self.pose_abs = torch.cat( [zeros,-0.9*ones,1.7*ones,-0.15*np.pi*ones,zeros,zeros], dim=1)
            self.dist = 1.7 / np.cos(0.15*np.pi) # this is to ease the generation of depth, so that depth can be zero meaned
            sensor_size = 32.
            focal_length = 60.
        elif self.opt.category == 'car1':
            self.pose_abs = torch.cat( [zeros,-ones, (np.sqrt(3)*ones).cuda(),(-np.pi/6.)*ones,zeros,zeros], dim=1)
            self.dist = np.sqrt(3) / np.cos(np.pi/6.) # this is to ease the generation of depth, so that depth can be zero meaned
            sensor_size = 32.
            focal_length = 60.
        elif self.opt.category == 'human':
            self.pose_abs = torch.cat( [zeros,zeros,1.4*ones,zeros,zeros,zeros], dim=1)
            self.dist = 1.4 # this is to ease the generation of depth, so that depth can be zero meaned
            sensor_size = 32.
            focal_length = 35.
        else:
            raise NotImplementedError("unknown category")

        intrinsics = np.array(
            [opt.fineSize / sensor_size * focal_length, 0., opt.fineSize / 2., \
             0., opt.fineSize / sensor_size * focal_length, opt.fineSize / 2., \
             0., 0., 1.]).reshape((3, 3))
        intrinsics_inv = np.linalg.inv(intrinsics)


        self.intrinsics = Variable(torch.from_numpy(intrinsics.astype(np.float32)).cuda()).unsqueeze(0).expand(opt.batchSize,3,3)
        self.intrinsics_inv = Variable(torch.from_numpy(intrinsics_inv.astype(np.float32)).cuda()).unsqueeze(0).expand(opt.batchSize,3,3)



        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)

        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        input_mask = input['mask']
        input_YawAB = input['YawAB']
        input_YawCB = input['YawCB']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
            input_mask = input_mask.cuda(self.gpu_ids[0], async=True)

            input_YawAB = input_YawAB.cuda(self.gpu_ids[0], async=True)
            input_YawCB = input_YawCB.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C
        self.input_mask = input_mask

        self.input_YawAB = input_YawAB
        self.input_YawCB = input_YawCB

        # print input_Yaw.size()

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        #
        if self.opt.category == 'car':

            self.maskB = torch.sum(self.input_B, dim=1)
            self.maskB = (self.maskB >= 3.0).unsqueeze(1)
            self.maskB = self.maskB.expand(self.input_B.size(0),3,self.input_B.size(2),self.input_B.size(3))
            #
            self.maskA = torch.sum(self.input_A, dim=1)
            self.maskA = (self.maskA >= 3.0).unsqueeze(1)
            self.maskA = self.maskA.expand(self.input_A.size(0),3,self.input_A.size(2),self.input_A.size(3))

            #
            self.maskC = torch.sum(self.input_C, dim=1)
            self.maskC = (self.maskC >= 3.0).unsqueeze(1)
            self.maskC = self.maskC.expand(self.input_C.size(0),3,self.input_C.size(2),self.input_C.size(3))



        if self.opt.category == 'human':

            self.maskB = torch.sum(self.input_B, dim=1)
            self.maskB = (self.maskB <= -3.0).unsqueeze(1)
            self.maskB = self.maskB.expand(self.input_B.size(0),3,self.input_B.size(2),self.input_B.size(3))
            #
            self.maskA = torch.sum(self.input_A, dim=1)
            self.maskA = (self.maskA <= -3.0).unsqueeze(1)
            self.maskA = self.maskA.expand(self.input_A.size(0),3,self.input_A.size(2),self.input_A.size(3))

            #
            self.maskC = torch.sum(self.input_C, dim=1)
            self.maskC = (self.maskC <= -3.0).unsqueeze(1)
            self.maskC = self.maskC.expand(self.input_C.size(0),3,self.input_C.size(2),self.input_C.size(3))



        self.maskB_fg = torch.sum(self.input_B, dim=1)
        self.maskB_fg = (self.maskB_fg < 3.0).unsqueeze(1)
        self.maskB_fg = self.maskB_fg.expand(self.input_B.size(0),3,self.input_B.size(2),self.input_B.size(3))
        #
        self.maskA_fg = torch.sum(self.input_A, dim=1)
        self.maskA_fg = (self.maskA_fg < 3.0).unsqueeze(1)
        self.maskA_fg = self.maskA_fg.expand(self.input_A.size(0),3,self.input_A.size(2),self.input_A.size(3))

        if self.opt.category == 'car1':
            return

        self.input_A[self.maskA] = 0.
        self.input_B[self.maskB] = 0.
        self.input_C[self.maskC] = 0.

    def forward(self):
        add_grid = self.opt.add_grid
        rectified = self.opt.rectified
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)
        self.real_mask = Variable(self.input_mask)

        self.real_YawAB= Variable(self.input_YawAB)
        self.real_YawCB = Variable(self.input_YawCB)

        b,c,h,w = self.real_A.size()
        zeros = Variable(torch.zeros((b,1)).cuda() )
        ones = Variable(torch.ones((b,1)).cuda() )

        pose_rel = torch.cat( [zeros,zeros,zeros,zeros,-self.real_YawCB, zeros], dim=1)

        R = rotation_tensor(zeros, self.real_YawAB, zeros).cuda()
        R_camera = rotation_tensor(np.pi/6.*ones, zeros, zeros).cuda()
        R_final =  R_camera.bmm(R.bmm(R_camera.transpose(1,2)))

        self.depth = self.netG(self.real_A, R_final)
        if self.opt.pred_mask:
            self.mask = F.tanh(self.depth[:,1,:,:]) * 1.5 - 0.5
            self.depth = self.depth[:,0,:,:].unsqueeze(1)
        self.depth = self.depth + self.dist

        self.fake_B_flow_converted = projection_layer2.inverse_warp(self.real_C, self.depth,
                                                                    pose_rel, self.pose_abs[:b,:], self.intrinsics[:b,:,:], self.intrinsics_inv[:b,:,:])

        if self.opt.pred_mask:
            self.fake_B = F.grid_sample(self.real_C, self.fake_B_flow_converted*self.real_mask.unsqueeze(-1).expand(b,h,w,2))
        else:
            self.fake_B = F.grid_sample(self.real_C, self.fake_B_flow_converted)

        if self.opt.use_masked_L1:
            self.fake_B = self.fake_B * self.real_mask
    # no backprop gradients
    def test(self):
        add_grid = self.opt.add_grid
        rectified = self.opt.rectified
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B_list = []

        NV = self.opt.test_views
        b = self.real_A.size(0)
        zeros = Variable(torch.zeros((b,1)).cuda() )

         #-4 / 9. * np.pi
        # yaw += 8 / 9. * np.pi / (NV - 1)
        yaw = 0
        real_A = self.real_A
        for i in range(NV):
            yaw += 2*np.pi/NV

            self.real_Yaw = Variable(-torch.Tensor([yaw]).cuda(self.gpu_ids[0], async=True)).unsqueeze(0)
            self.depth = self.netG(real_A, self.real_Yaw) + self.dist
            pose_rel = torch.cat([zeros, zeros, zeros, zeros, -self.real_Yaw, zeros], dim=1)

            self.fake_B_flow_converted = projection_layer2.inverse_warp(real_A, self.depth,
                                                                        pose_rel, self.pose_abs[:b,:],
                                                                        self.intrinsics[:b, :, :],
                                                                        self.intrinsics_inv[:b, :, :])
            self.fake_B = F.grid_sample(real_A, self.fake_B_flow_converted)
            self.fake_B_list.append(self.fake_B)

            if np.mod(i, NV/4) == 0:
                print 'hi'
                real_A = self.fake_B
                yaw =0


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        self.loss_TV = self.criterionTV(self.depth) * self.opt.lambda_tv
        if self.opt.pred_mask:
            self.loss_mask = self.criterionL1(self.mask,self.real_mask) * self.opt.lambda_mask
        else:
            self.loss_mask = 0. * self.loss_TV


        if self.opt.lambda_depth > 0:
            self.loss_G_depth = self.criterionL1(self.depth[self.maskB[:,:1,:,:]],
                                                Variable(-1*torch.ones(self.depth.size()).cuda())[self.maskB[:,:1,:,:]]) * self.opt.lambda_depth
        else:
            self.loss_G_depth = 0. * self.loss_TV

        # Second, G(A) = B
        self.loss_G_L1_masked = self.criterionL1(self.fake_B[self.maskB_fg], self.real_B[self.maskB_fg]) * self.opt.lambda_A
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A



        self.loss_G = self.loss_G_L1 + self.loss_TV + self.loss_G_depth + self.loss_mask #+ self.loss_kl #+ self.loss_G_L1_masked

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_L1', self.loss_G_L1.data[0]),
                            ('G_L1_masked', self.loss_G_L1_masked.data[0]),
                            ('F_L1', self.loss_G_depth.data[0]),
                            ('TV', self.loss_TV.data[0]),
                            ])

    def get_current_visuals(self):
        if not self.opt.isTrain:
            return self.get_current_visuals_test()
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_C = util.tensor2im(self.real_C.data)

        flow = util.tensor2im(self.fake_B_flow_converted.permute(0,3,1,2).data)
        depth = util.tensor2im(self.depth.data)

        if self.opt.pred_mask:
            mask = util.tensor2im(self.mask.data)
            real_mask = util.tensor2im(self.real_mask.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('real_C', real_C),
                            ('flow', flow), ('depth', depth), ('mask', mask), ('real_mask', real_mask)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),('real_C', real_C),
                            ('flow',flow), ('depth', depth)])

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

#
# if self.opt.lambda_depth > 0:
#     self.loss_G_depth = self.criterionL1(self.depth[self.maskB[:,:1,:,:]],
#                                         Variable(-1*torch.ones(self.depth.size()).cuda())[self.maskB[:,:1,:,:]]) * self.opt.lambda_depth
# else:
#     self.loss_G_depth = 0. * self.loss_TV

# KL loss
# if self.opt.lambda_kl > 0:
#     kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
#     self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
# else:
#     self.loss_kl = 0. * self.loss_TV