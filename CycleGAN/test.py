import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from data import create_dataset
from models import create_model

from util.util import save_images


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot',
                        default='./datasets/nature_image',
                        help='directory of images to be transferred',
                        type=str)
    parser.add_argument('--result_dir',
                        default='./results',
                        help='directory to save images after style transfer',
                        type=str)
    parser.add_argument('--load_size',
                        default=512,
                        help='load size of test images',
                        type=int)
    parser.add_argument('--crop_size',
                        default=512,
                        help='crop size of test images',
                        type=int)
    parser.add_argument('--name',
                        default='nature2painting',
                        help='name of experiment, also where model is stored',
                        type=str)

    args = parser.parse_args()
    
    dataroot = args.dataroot
    results_dir = args.result_dir
    load_size = args.load_size
    crop_size = args.crop_size
    name = args.name
    
    
    dataset = create_dataset(dataroot, batch_size=1, phase='test', load_size=load_size, crop_size=crop_size, serial_batches=True, input_nc=3, output_nc=3, no_flip = True)
    model = create_model(isTrain=False, name=name, model='test')
    model.setup()


    for i, data in enumerate(tqdm(dataset)):

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths() 

        save_images(visuals, results_dir, img_path, name=str(i))
