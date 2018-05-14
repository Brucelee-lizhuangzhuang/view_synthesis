import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import torch.nn.functional as F
import itertools
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
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, nz * 3)]) #, nn.LeakyReLU(0.2, True),
        if use_vae:
            self.fc_var = nn.Sequential(*[nn.Linear(ndf * nf_mult, nz * 3), nl_layer_enc()])
        self.fc2 = nn.Sequential(*[nn.Linear(nz * 3, ndf * nf_mult),nl_layer_enc()]) #, nn.BatchNorm1d(ndf * nf_mult)

        deconv = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 4)

            upsample = 'bilinear' if n_layers - n < n_bilinear_layers else 'basic'
            deconv += networks.upsampleLayer(ndf * nf_mult_prev, ndf * nf_mult, upsample=upsample)
            if norm_layer is not None and (n_layers - n + 1) < n_layers:
                deconv += [norm_layer(ndf * nf_mult)]
            deconv += [nl_layer_dec()]

        if n_bilinear_layers > 0:
            deconv += networks.upsampleLayer(ndf, 2, upsample='bilinear')
        else:
            deconv += networks.upsampleLayer(ndf, 2, upsample='basic')

        # deconv += [nn.Tanh()]

        self.deconv = nn.Sequential(*deconv)
        self.nz = nz

    def forward(self, x, R, Tz=0 ):
        z_conv = self.enc(x)

        z_fc = self.fc(z_conv.view(x.size(0),-1) ).view(x.size(0), self.nz, 3)
        z_fc = F.tanh(z_fc)

        z_rot = z_fc.bmm(R) # + T
        z_rot_fc = self.fc2(z_rot.view(x.size(0), self.nz*3))

        output = self.deconv(z_rot_fc.view(z_conv.size(0),z_conv.size(1),z_conv.size(2),z_conv.size(3)))

        return output


    def get_mu_var(self):
        return self.mu,self.logvar


class MultiViewFlowModel(BaseModel):
    def name(self):
        return 'MultiViewFlowModel'

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

        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):

        self.real_A = Variable(self.input_A,volatile=True)
        self.real_B = Variable(self.input_B,volatile=True)
        self.real_C = Variable(self.input_C,volatile=True)
        self.real_mask = Variable(self.input_mask,volatile=True)

        self.real_YawAB= Variable(self.input_YawAB,volatile=True)
        self.real_YawCB = Variable(self.input_YawCB,volatile=True)

        b,c,h,w = self.real_A.size()
        zeros = Variable(torch.zeros((b,1)).cuda() )

        R = rotation_tensor(zeros, zeros, self.real_YawAB).cuda()
        R_final = R #R_camera.bmm(R.bmm(R_camera.transpose(1,2)))

        self.flow = self.netG(self.real_A, R_final).permute(0,2,3,1)
        self.flow_converted = self.flow + self.grid[:b,:,:,:]

        self.fake_B = F.grid_sample(self.real_C, self.flow_converted)

    # no backprop gradients
    def test(self):
        if self.opt.list_path is not None:
            self.test_list()
            return
        if self.opt.auto_aggressive:
            self.test_auto_aggressive()
        else:
            self.test_normal()

    def test_list(self):

        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.real_B = Variable(self.input_B, volatile=True)
            self.real_C = Variable(self.input_C, volatile=True)
            self.real_mask = Variable(self.input_mask, volatile=True)
            self.real_YawAB = Variable(self.input_YawAB, volatile=True)
            self.real_YawCB = Variable(self.input_YawCB, volatile=True)

            b,c,h,w = self.real_A.size()
            zeros = Variable(torch.zeros((b,1)).cuda(),volatile=True )

            pose_rel = torch.cat([zeros, zeros, zeros, zeros, -self.real_YawCB, zeros], dim=1)

            R = rotation_tensor(zeros, zeros, self.real_YawAB).cuda()
            # R_camera = rotation_tensor(np.pi/6*ones, zeros,zeros).cuda()
            R_final = R  # R_camera.bmm(R.bmm(R_camera.transpose(1,2)))

            self.depth = self.netG(self.real_A, R_final)
            self.depth = self.depth + self.dist

            self.fake_B_flow_converted = projection_layer2.inverse_warp(self.real_A, self.depth,
                                                                        pose_rel, self.pose_abs[:b, :],
                                                                        self.intrinsics[:b, :, :],
                                                                        self.intrinsics_inv[:b, :, :])

            self.fake_B = F.grid_sample(self.real_A, self.fake_B_flow_converted)
            self.loss_G_L1 = torch.nn.L1Loss()(self.fake_B[self.real_mask.unsqueeze(1).expand(b,3,h,w)], self.real_B[self.real_mask.unsqueeze(1).expand(b,3,h,w)])
    def get_errors(self):
        return self.loss_G_L1

    def test_normal(self):

        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.real_B = Variable(self.input_B, volatile=True)
            self.fake_B_list = []

            NV = self.opt.test_views
            b,c,h,w = self.real_A.size()
            zeros = Variable(torch.zeros((b,1)).cuda(),volatile=True )

            if self.opt.only_neighbour:
                whole_range = 4*np.pi/9.  # 80
                yaw = -2*np.pi/9. # -40
            else:
                whole_range = 2*np.pi
                yaw = 0

            if self.opt.category == 'human':
                whole_range = np.pi  # 80
                yaw = -np.pi/2. # -40

            yaw -= whole_range / NV
            real_A = self.real_A
            for i in range(NV):
                yaw += whole_range/NV

                self.real_Yaw = Variable(-torch.Tensor([yaw]).cuda(self.gpu_ids[0], async=True) ,volatile=True).unsqueeze(0)
                pose_rel = torch.cat([zeros, zeros, zeros, zeros, -self.real_Yaw, zeros], dim=1)

                R = rotation_tensor(zeros, zeros, self.real_Yaw).cuda()
                R_final = R #R_camera.bmm(R.bmm(R_camera.transpose(1, 2)))

                self.depth = self.netG(real_A, R_final)
                self.depth = self.depth + self.dist

                self.fake_B_flow_converted = projection_layer2.inverse_warp(real_A, self.depth,
                                                                            pose_rel, self.pose_abs[:b, :],
                                                                            self.intrinsics[:b, :, :],
                                                                            self.intrinsics_inv[:b, :, :])

                self.fake_B = F.grid_sample(real_A, self.fake_B_flow_converted)

                self.fake_B_list.append(self.fake_B)


    def test_auto_aggressive(self):

        with torch.no_grad():
            self.real_A = Variable(self.input_A, volatile=True)
            self.real_B = Variable(self.input_B, volatile=True)
            self.fake_B_list = []

            NV = self.opt.test_views
            b, c, h, w = self.real_A.size()
            zeros = Variable(torch.zeros((b, 1)).cuda(), volatile=True)

            whole_range = 4 * np.pi / 9.  # 80
            yaw = -2 * np.pi / 9.  # -40


            whole_range /= 2  # 80
            NV /= 2
            yaw = 0

            yaw -= whole_range / NV
            real_A = self.real_A
            for i in range(NV):
                yaw += whole_range / NV

                self.real_Yaw = Variable(-torch.Tensor([yaw]).cuda(self.gpu_ids[0], async=True),
                                         volatile=True).unsqueeze(0)
                pose_rel = torch.cat([zeros, zeros, zeros, zeros, -self.real_Yaw, zeros], dim=1)

                R = rotation_tensor(zeros, zeros, self.real_Yaw).cuda()
                R_final = R  # R_camera.bmm(R.bmm(R_camera.transpose(1, 2)))

                self.depth = self.netG(real_A, R_final)
                self.depth = self.depth + self.dist

                self.fake_B_flow_converted = projection_layer2.inverse_warp(real_A, self.depth,
                                                                            pose_rel, self.pose_abs[:b, :],
                                                                            self.intrinsics[:b, :, :],
                                                                            self.intrinsics_inv[:b, :, :])
                self.fake_B = F.grid_sample(real_A, self.fake_B_flow_converted)

                self.fake_B_list.append(self.fake_B)

                if np.mod(i, 10) == 0:
                    real_A = self.fake_B
                    yaw = 0

            fake_B_list_reverse = []


            yaw = whole_range / NV
            real_A = self.real_A
            for i in range(NV):
                yaw -= whole_range / NV

                self.real_Yaw = Variable(-torch.Tensor([yaw]).cuda(self.gpu_ids[0], async=True),
                                         volatile=True).unsqueeze(0)
                pose_rel = torch.cat([zeros, zeros, zeros, zeros, -self.real_Yaw, zeros], dim=1)

                R = rotation_tensor(zeros, zeros, self.real_Yaw).cuda()
                R_final = R  # R_camera.bmm(R.bmm(R_camera.transpose(1, 2)))

                self.depth = self.netG(real_A, R_final)
                self.depth = self.depth + self.dist

                self.fake_B_flow_converted = projection_layer2.inverse_warp(real_A, self.depth,
                                                                            pose_rel, self.pose_abs[:b, :],
                                                                            self.intrinsics[:b, :, :],
                                                                            self.intrinsics_inv[:b, :, :])
                self.fake_B = F.grid_sample(real_A, self.fake_B_flow_converted)

                fake_B_list_reverse.append(self.fake_B)

                if np.mod(i, 10) == 0:
                    real_A = self.fake_B
                    yaw = 0
            self.fake_B_list =  list(reversed(fake_B_list_reverse)) + self.fake_B_list
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        self.loss_TV = self.criterionTV(self.flow) * self.opt.lambda_tv

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_L1 + self.loss_TV

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_L1', self.loss_G_L1.data[0]),
                            ('TV', self.loss_TV.data[0]),
                            ])

    # ('G_L1_masked', self.loss_G_L1_masked.data[0]),

    def get_current_visuals(self):
        if not self.opt.isTrain:
            if self.opt.list_path is None:
                return self.get_current_visuals_test()
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_C = util.tensor2im(self.real_C.data)

        flow = util.tensor2im(self.flow.permute(0,3,1,2).data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),('real_C', real_C),
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
