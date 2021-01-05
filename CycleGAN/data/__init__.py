"""This package includes all the modules related to data loading and preprocessing

"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def create_dataset(dataroot, batch_size=1, phase='train', load_size=286, crop_size=256, serial_batches=False, input_nc=3, output_nc=3, no_flip=True):
    """Create a dataset given the option.

    """
    data_loader = CustomDatasetDataLoader(dataroot, batch_size=batch_size, phase=phase, load_size=load_size, crop_size=crop_size, serial_batches=serial_batches, input_nc=input_nc, output_nc=output_nc, no_flip=no_flip)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, dataroot, batch_size=1, phase='train', load_size=286, crop_size=256, serial_batches=False, input_nc=3, output_nc=3, no_flip=True):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.batch_size = batch_size

        dataset_name = "unaligned" if phase=='train' else "single"

        dataset_filename = "data." + dataset_name + "_dataset"
        datasetlib = importlib.import_module(dataset_filename)

        dataset = None
        target_dataset_name = dataset_name.replace('_', '') + 'dataset'
        for name, cls in datasetlib.__dict__.items():
            if name.lower() == target_dataset_name.lower() \
               and issubclass(cls, BaseDataset):
                dataset = cls

        dataset_class = dataset

        self.dataset = dataset_class(dataroot, phase=phase, load_size=load_size, crop_size=crop_size, serial_batches=serial_batches, input_nc=input_nc, output_nc=output_nc, no_flip=no_flip)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
            )

    def load_data(self):
        return self

    def __len__(self):
        """
        Return the number of data in the dataset
        
        """
        return len(self.dataset)

    def __iter__(self):
        """
        Return a batch of data
        
        """
        for i, data in enumerate(self.dataloader):
            yield data
