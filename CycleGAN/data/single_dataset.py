from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, dataroot, batch_size=1, phase='test', load_size=512, crop_size=512, serial_batches=True, input_nc=3, output_nc=3, no_flip=True):

        BaseDataset.__init__(self, dataroot, load_size=load_size, crop_size=load_size, no_flip=no_flip)
        self.A_paths = sorted(make_dataset(dataroot))
        self.transform = get_transform(no_flip=True, load_size=load_size, crop_size=crop_size, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
