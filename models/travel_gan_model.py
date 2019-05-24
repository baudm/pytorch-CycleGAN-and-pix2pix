import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import itertools

from torchsummary import summary

from .base_model import BaseModel
from . import networks


class TravelGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(
            netG='unet_128', ndf=40, norm='batch',
            dataset_mode='celeba', batch_size=32
        )
        if is_train:
            parser.set_defaults(
                lr=0.0002, beta1=0.5,
                gan_mode='lsgan'
            )
            parser.add_argument('--beta2', type=float, default=0.9, help='second momentum term of adam')
            parser.add_argument('--siamese_margin', type=float, default=50.0, help='Siamese contrastive loss margin')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G_adv', 'TraVeL', 'Sc']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D', 'S']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        self.netG = networks.define_G(opt.input_nc + 2, opt.output_nc + 2, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        summary(self.netG, (opt.input_nc + 2, opt.crop_size, opt.crop_size))

        if self.isTrain:  # define discriminators
            # self.netD = networks.define_D_travel(opt.input_nc, opt.ndf, 1, True, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = networks.define_D(opt.input_nc + 2, 64, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netS = networks.define_D_travel(opt.input_nc + 2, opt.ndf, 1000, False, opt.init_type, opt.init_gain, self.gpu_ids)
            summary(self.netD, (opt.input_nc + 2, opt.crop_size, opt.crop_size))
            summary(self.netS, (opt.input_nc + 2, opt.crop_size, opt.crop_size))

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionDist = lambda v1, v2: 1. - F.cosine_similarity(v1, v2, dim=-1)
            self.criterionSiamese = lambda v: torch.clamp(opt.siamese_margin - torch.norm(v, dim=-1), min=0.)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = optim.Adam(itertools.chain(self.netG.parameters(), self.netS.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # A - no hats, B - with hats
        AtoB = self.opt.direction == 'BtoA'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

    def cat_coords(self, img):
        bs = img.shape[0]
        w, h = img.shape[-2:]
        x = torch.linspace(-1, 1, w).to(self.device)
        y = torch.linspace(-1, 1, h).to(self.device)
        x, y = torch.meshgrid(x, y)
        # Add two dimensions, tile so that shape is: (bs, 1, w, h)
        x = x[2 * (None,)].repeat((bs, 1, 1, 1))
        y = y[2 * (None,)].repeat((bs, 1, 1, 1))
        # Concat channel dimensions
        return torch.cat([img, x, y], dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # CoordConv
        self.fake_B_coord = self.netG(self.cat_coords(self.real_A))
        # image without coordinates
        self.fake_B = self.fake_B_coord[:, :3, :, :]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = self.netD(self.cat_coords(self.real_B))
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD(self.fake_B_coord.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate the loss for generator and siamese network"""
        bs = len(self.real_A)

        # Get embeddings of A and G(A)
        s_A = self.netS(self.cat_coords(self.real_A))
        s_B = self.netS(self.fake_B_coord)

        # Tile
        s_A_i = s_A.repeat([bs - 1, 1])
        s_B_i = s_B.repeat([bs - 1, 1])

        # Tile in permuted order
        i = torch.arange(bs)
        idx = torch.cat([i.roll(s) for s in range(1, bs)])
        s_A_j = s_A[idx]
        s_B_j = s_B[idx]

        # Get transformation vectors
        v_A = s_A_i - s_A_j
        v_B = s_B_i - s_B_j

        # Transformation Vector loss
        self.loss_TraVeL = self.criterionDist(v_A, v_B).sum()

        # Siamese contrastive loss
        self.loss_Sc = self.criterionSiamese(v_A).sum()

        # GAN loss D(G(A))
        self.loss_G_adv = self.criterionGAN(self.netD(self.fake_B_coord), True)

        loss_S = self.loss_Sc + self.loss_TraVeL
        loss_G = self.loss_G_adv + self.loss_TraVeL
        loss_total = loss_S + loss_G
        loss_total.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
