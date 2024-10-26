from torchvision import transforms
from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import scipy.io as scio
import numpy as np
import pdb
import os

class MyToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return TF.to_tensor(pic.copy())

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Dataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False, frame_num=-1):
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        self.frame_num = frame_num
        self.verbose = verbose
        for root, _, files in os.walk(self.root_dirs[0]):
            for file in files:
                self.images_path.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]
        image = np.fromfile(img_name, np.float32).reshape(128, 128, 8, 24)
        image = torch.from_numpy(image.transpose(3, 2, 0, 1))
        if self.verbose:
            return image, img_name
        return image

def get_gt(img_path):
    tfs = []
    tfs += [
    MyToTensor()
    ]
    gt_transforms = transforms.Compose(tfs)
    image = np.fromfile(img_path.replace('noise', 'clean'), np.float32).reshape(128, 128, 8, 24)
    image = gt_transforms(image)

    return image

def get_dataloaders(test_path_list,
                    drop_last=True, n_worker=0, verbose=False, frame_num=-1):
    batch_sizes = {'test':1}
    test_transforms = transforms.Compose([MyToTensor()])
    data_transforms = {'test': test_transforms}
    image_datasets = {'test': Dataset(test_path_list, data_transforms['test'], verbose=verbose, frame_num=frame_num)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                  num_workers=n_worker, drop_last=drop_last, shuffle=False)
                   for x in ['test']}
    return dataloaders
