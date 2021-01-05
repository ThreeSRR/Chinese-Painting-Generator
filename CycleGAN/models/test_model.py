from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):

    def __init__(self, gpu_ids='0', isTrain=False, checkpoints_dir='./checkpoints', name='experiment_name', continue_train=False, model='cycle_gan'):
        """Initialize the pix2pix class."""
        
        assert(not isTrain)
        BaseModel.__init__(self, gpu_ids=gpu_ids, isTrain=isTrain, checkpoints_dir=checkpoints_dir, name=name, continue_train=continue_train, verbose=False)

        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64 # num of gen filters in the last conv layer
        self.ndf = 64 # num of discriminator filters in the first conv layer'
        self.netG = 'resnet_9blocks' # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.norm = 'instance' # instance normalization or batch normalization [instance | batch | none]
        self.no_dropout = True
        self.init_type = 'normal' # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.02
        self.netD = 'basic' # specify discriminator architecture [basic | n_layers | pixel]
        self.n_layers_D = 3 # only used if netD==n_layers
        self.pool_size = 50 # the size of image buffer that stores previously generated images
        self.lr = 0.0002
        self.beta1 = 0.5 # momentum term of adam
        self.gan_mode = 'lsgan' # the type of GAN objective. [vanilla| lsgan | wgangp]
        self.model_suffix = ''

        self.loss_names = []
        self.visual_names = ['real', 'fake']
        self.model_names = ['G' + self.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(self.input_nc, self.output_nc, self.ngf, self.netG,
                                      self.norm, not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)

        setattr(self, 'netG' + self.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):

        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
