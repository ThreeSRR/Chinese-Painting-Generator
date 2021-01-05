import torch
import numpy as np
from PIL import Image
import os


def mkdirs(paths):

    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(input_image, imtype=np.uint8):

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    image_pil.save(image_path)


def save_images(visuals, image_dir, image_path, name):
    
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    
    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path)


def print_current_losses(epoch, iters, losses, t_comp, t_data, log_name='train_log'):

    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)