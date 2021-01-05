import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from models import networks


class CycleGANModel(BaseModel):
    """Implement of the CycleGAN model"""
    
    def __init__(self, gpu_ids='0', isTrain=True, checkpoints_dir='./checkpoints', name='experiment_name', continue_train=False, preprocess='resize_and_crop', verbose=False):

        self.lambda_A = 10.0 # weight for cycle loss (A -> B -> A)
        self.lambda_B = 10.0 # weight for cycle loss (B -> A -> B)
        self.lambda_identity = 0.5 # 'use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1'

        BaseModel.__init__(self, gpu_ids, isTrain, checkpoints_dir, name, continue_train, preprocess, verbose)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # params of networks
        
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64 # num of gen filters in the last conv layer
        self.ndf = 64 # num of discriminator filters in the first conv layer'
        self.netG = 'resnet_9blocks' # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.norm = 'instance' # instance normalization or batch normalization [instance | batch | none]
        self.no_dropout = False
        self.init_type = 'normal' # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.02
        self.netD = 'basic' # specify discriminator architecture [basic | n_layers | pixel]
        self.n_layers_D = 3 # only used if netD==n_layers
        self.pool_size = 50 # the size of image buffer that stores previously generated images
        self.lr = 0.0002
        self.beta1 = 0.5 # momentum term of adam
        self.gan_mode = 'lsgan' # the type of GAN objective. [vanilla| lsgan | wgangp]

        
        # define networks
        self.netG_A = networks.define_G(self.input_nc, self.output_nc, self.ngf, self.netG, self.norm,
                                        self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(self.output_nc, self.input_nc, self.ngf, self.netG, self.norm,
                                        self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(self.output_nc, self.ndf, self.netD,
                                            self.n_layers_D, self.norm, self.init_type, self.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(self.input_nc, self.ndf, self.netD,
                                            self.n_layers_D, self.norm, self.init_type, self.init_gain, self.gpu_ids)

        if self.isTrain:
            if self.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(self.input_nc == self.output_nc)
            self.fake_A_pool = ImagePool(self.pool_size)
            self.fake_B_pool = ImagePool(self.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(self.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        AtoB = True
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        """Run forward pass"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)


    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()


    def optimize_parameters(self):

        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
