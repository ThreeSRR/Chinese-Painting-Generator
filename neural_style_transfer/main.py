"""
Neural Transfer Using PyTorch

The algorithm takes three images,
an input image, a content-image, and a style-image, and changes the input 
to resemble the content of the content-image and the artistic style of the style-image.

"""


import copy
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from LossFunction import ContentLoss, StyleLoss


def image_loader(image_name, transform, device):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)

    return image.to(device, torch.float)


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers, style_layers, device):
    '''
    to add content loss and style loss layers after convolution layer by creating a new Sequential module

    '''
    cnn = copy.deepcopy(cnn)

    content_loss_list = []
    style_loss_list = []

    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_loss_list.append(content_loss)

        if name in style_layers:
            # add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_loss_list.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_loss_list, content_loss_list


def get_input_optimizer(input_img):
    '''
    L-BFGS algorithm to run our gradient descent
    to train the input image in order to minimise the content/style losses
    '''
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, content_layers_default, style_layers_default, content_img, style_img, input_img, device, 
                       num_steps=300, style_weight=10000, content_weight=1):
    """
    the function to perform neural transfer
    
    """
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img, 
        content_layers_default, style_layers_default, device)
    optimizer = get_input_optimizer(input_img)

    epoch = [0]
    while epoch[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print("epoch {}:  Style Loss : {:4f} Content Loss: {:4f}".format(epoch[0], style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    
    return input_img


def main(style_img, content_img, outputpath='./result.png', num_steps=500, style_weight=100000, content_weight=1):
    '''
    the main function of neural style transfer

    :param style_img: the image with target style you want to transfer to
    :param content_img: the original image, to transfer its style while reserve its content
    :param outputpath: the path to save image with transferred style
    :param num_steps:  number of steps to update parameters
    :param style_weight: weight of style
    :param content_weight: weight of loss
    '''

    imsize = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()
    ])

    style_img = image_loader(style_img, transform, device)
    content_img = image_loader(content_img, transform, device)

    # use the features module of pretrained vgg19
    # need the output of the individual convolution layers to measure content and style loss.
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    input_img = content_img.clone()

    output = run_style_transfer(cnn, content_layers_default, style_layers_default, content_img, style_img, input_img, device, 
                                num_steps=num_steps, style_weight=style_weight, content_weight=content_weight)
    output = output.detach().cpu().numpy().squeeze(0).transpose([1,2,0])
    plt.imsave(outputpath, output)


if __name__ == '__main__':

    style_img_path = "./data/style.jpg"
    content_img_path = "./data/content2.jpg"

    main(style_img_path, content_img_path, outputpath='./result/result2.png', num_steps=500, style_weight=100000, content_weight=1)
