import time
import os
import argparse
from data import create_dataset
from models import create_model
from util.util import mkdirs, print_current_losses


def main(dataroot, checkpoints_dir, name, n_epochs, n_epochs_decay, gpu_ids):

    epoch_count = 1
    batch_size = 1
    isTrain = True

    expr_dir = os.path.join(checkpoints_dir, name)
    mkdirs(expr_dir)
    print_freq = 100
    save_epoch_freq = 1


    dataset = create_dataset(dataroot, batch_size=batch_size, phase='train', load_size=286, crop_size=256, serial_batches=False, input_nc=3, output_nc=3)
    dataset_size = len(dataset)

    model = create_model(gpu_ids=gpu_ids, isTrain=isTrain, checkpoints_dir=checkpoints_dir, name=name, continue_train=False)
    
    model.setup()
    
    total_iters = 0

    for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.update_learning_rate()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += batch_size
            epoch_iter += batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            iter_data_time = time.time()
        if epoch % save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot',
                        default='./datasets/chinesepainting',
                        help='directory of training data',
                        type=str)
    parser.add_argument('--n_epochs',
                        default=200,
                        help='number of epochs',
                        type=int)
    parser.add_argument('--n_epochs_decay',
                        default=200,
                        help='number of epochs where learning rate decays',
                        type=int)
    parser.add_argument('--gpu_id',
                        default='0',
                        help='gpu_id, -1 if use cpu',
                        type=str)
    parser.add_argument('--checkpoints_dir',
                        default='./checkpoints',
                        help='directory to save checkpoints',
                        type=str)
    parser.add_argument('--name',
                        default='chinesepainting_cyclegan',
                        help='name of experiment',
                        type=str)

    args = parser.parse_args()

    dataroot = args.dataroot
    n_epochs = args.n_epochs
    n_epochs_decay = args.n_epochs_decay
    gpu_ids = args.gpu_id
    checkpoints_dir = args.checkpoints_dir
    name = args.name
    
    main(dataroot, checkpoints_dir, name, n_epochs, n_epochs_decay, gpu_ids)
    