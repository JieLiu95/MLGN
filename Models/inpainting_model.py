import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from .vgg import VGG, GramMatrix, GramMSELoss
from . import models_gd
import pytorch_msssim


class InpaintingModel(BaseModel):
    def name(self):
        return 'InpaintingModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc + 1,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.y = self.Tensor(opt.batchSize)
        self.opt = opt

        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.loss_layers = self.style_layers
        self.loss_fns = [GramMSELoss()] * len(self.style_layers)
        if torch.cuda.is_available():
            self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]

        self.loss_ms_ssim = pytorch_msssim.MS_SSIM(data_range=1)
        if torch.cuda.is_available():
            self.loss_ms_ssim.cuda()

        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(os.getcwd() + '/Models/' + 'vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()

        print(self.vgg.state_dict().keys())

        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.weights = self.style_weights

        # load/define networks

        self.netG = models_gd.Generator().cuda()  # first stage
        networks.print_network(self.netG)
        print(self.netG)
        weights = torch.load(os.getcwd() + '/checkpoints/Inpainting/' + 'celeba-hq-random_net_G.pth')

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('------------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('------------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        # generate mask, 1 represents masked point
        # mask: mask region {0, 1}
        # x_incomplete: incomplete image, [-1, 1]
        # returns: [-1, 1] as predicted image
        self.bbox = util.random_bbox(self.opt)
        self.mask = util.bbox2mask(self.bbox, self.opt)  # Tensor
        self.x_incomplete = input_A * (1.-self.mask)  # can broadcast
        ones_x = torch.zeros_like(self.x_incomplete)[:, 0:1, :, :]
        x = torch.cat((self.x_incomplete, ones_x*self.mask), 1)
        self.input_A.resize_(x.size()).copy_(x)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = input['A_paths']
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
            self.x_incomplete = self.x_incomplete.cuda()

    def set_input_test(self, input):
        input_A = input['A']
        # input_B = input['B']
        # generate mask, 1 represents masked point
        # mask: mask region {0, 1}
        # x_incomplete: incomplete image, [-1, 1]
        # returns: [-1, 1] as predicted image
        # self.mask = util.test_mask(186, 205, 94, 158)  # Center $ Tensor
        self.bbox = util.random_bbox(self.opt)
        self.mask = util.bbox2mask(self.bbox, self.opt)  # Random & Tensor
        self.x_incomplete = input_A * (1.-self.mask)  # can broadcast
        ones_x = torch.zeros_like(self.x_incomplete)[:, 0:1, :, :]
        x = torch.cat((self.x_incomplete, ones_x*self.mask), 1)
        self.input_A.resize_(x.size()).copy_(x)
        # self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = input['A_paths']
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
            self.x_incomplete = self.x_incomplete.cuda()

    def forward(self):
        self.real_A = Variable(self.input_A)   # channel 4
        self.mask_variable = Variable(self.mask, requires_grad=False)
        self.x_incomplete_variable = Variable(self.x_incomplete, requires_grad=False)
        self.fake_B = self.netG.forward(self.real_A) * self.mask_variable + self.x_incomplete_variable * (1.-self.mask_variable)

        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.mask_variable = Variable(self.mask, volatile=True)
        self.x_incomplete_variable = Variable(self.x_incomplete, volatile=True)
        self.fake_B = self.netG.forward(self.real_A) * self.mask_variable + self.x_incomplete_variable * (1.-self.mask_variable)
        # self.real_B = Variable(self.input_B, volatile=True)
        self.temp = self.netG.forward(self.real_A).data

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_B.clone()
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = self.real_B.clone()
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        if self.opt.use_style:
            style_targets = [GramMatrix()(A).detach() for A in self.vgg(self.real_B, self.style_layers)]  # gram matrices
            targets = style_targets
            out = self.vgg(self.fake_B, self.loss_layers)
            layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
            loss = 5 * sum(layer_losses)
            self.style_loss = loss
            loss.backward(retain_graph=True)
            self.style_loss_value = self.style_loss.data[0]
        else:
            self.style_loss_value = 0

        self.loss_ms_ssim_value = 5 * (1 - self.loss_ms_ssim((self.fake_B + 1) * 0.5, (self.real_B + 1) * 0.5))

        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B.clone()
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B * self.mask_variable, self.real_B * self.mask_variable) * self.opt.lambda_l1  # lambda=100

        self.loss_G = 5 * self.loss_G_GAN + self.loss_G_L1 + self.loss_ms_ssim_value  # face: 5, 1, 1; natural: 1, 10, 5

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
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ('Style', self.style_loss_value),
                            ('MS-SSIM', self.loss_ms_ssim_value)
                            ])

    def get_current_visuals(self):

        if self.isTrain:
            real_A = util.tensor2im(self.x_incomplete_variable.data)
            fake_B = util.tensor2im(self.fake_B.data)
            real_B = util.tensor2im(self.real_B.data)
            mask = util.tensor2im_0(self.mask_variable.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('mask', mask)])  # for test
        else:
            real_A = util.tensor2im(self.x_incomplete_variable.data)
            fake_B = util.tensor2im(self.fake_B.data)
            mask = util.tensor2im_0(self.mask_variable.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('mask', mask)])  # for test

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
