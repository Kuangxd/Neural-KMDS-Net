from torchvision import transforms
from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import random
from typing import Sequence
from itertools import repeat
import scipy.io as scio
import numpy as np
import torch
import re
import pdb
import os

np_str_obj_array_pattern = re.compile(r'[SaUO]')
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data
class MyResize:
    def __init__(self, scal_ratio,crop):
        self.scal_ratio = scal_ratio
        self.crop = crop
        


    def __call__(self, x):
        # bands = x.shape[2]
        # if bands > 31:
        #     bs = int(np.random.rand(1) * bands)
        #     if bs + 31 > bands:
        #         bs = bands - 31
        #     x = x[:, :, bs:bs + 31]
        # pdb.set_trace()
        scale =np.random.choice(self.scal_ratio)
        im_sz=x.shape
        rs=[int(im_sz[0]*scale),int(im_sz[1]*scale)]
        if rs[0]<self.crop:
            rs[0]=self.crop
        if rs[1] < self.crop:
            rs[1] = self.crop

        im = np.zeros([rs[0], rs[1], im_sz[2]],dtype=x.dtype)
        for i in range(im_sz[2]):
            im[:,:,i]=np.array(Image.fromarray(x[:,:,i]).resize(rs)).T
        # print(scale)
        # im= TF.resize(x,self.sizes)
        return im

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
class MyRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.flipud(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
class MyRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.fliplr(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
class MyRandomCrop(object):
      def __init__(self, size):
        self.size=size

      def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        _w, _h, _b = img.shape
        x = random.randint(1, _w)
        y = random.randint(1, _h)
        x2 = x + self.size
        y2 = y + self.size
        if x2 > _w:
            x2 = _w
            x = _w - self.size
        if y2 > _h:
            y2 = _h
            y = _h - self.size
        cropImg = img[(x):(x2), (y):(y2), :]
        # print(x, x2, y, y2)
        # pdb.set_trace()
        return cropImg

        # return self.cropit(img,self.size)
        # return img
      def cropit(image, crop_size):
          _w, _h, _b = image.shape
          x = random.randint(1, _w)
          y = random.randint(1, _h)
          x2 = x + crop_size
          y2 = y + crop_size
          if x2 > _w:
              x2 = _w
              x = _w - crop_size
          if y2 > _h:
              y2 = _h
              y = _h - crop_size
          cropImg = image[(x):(x2), (y):(y2), :]
          return cropImg
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
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []

        for root, _, files in os.walk(os.path.join(root_dirs[0], 'clean')):
            for file in files:
                self.images_path.append(os.path.join(root, file))        

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_clean_path = self.images_path[idx]
        img_noise_path = img_clean_path.replace('clean', 'noise')
        
        image_clean = np.fromfile(img_clean_path, np.float32).reshape(128, 128, 8, 24).transpose(3, 2, 0, 1)
        image_noise = np.fromfile(img_noise_path, np.float32).reshape(128, 128, 8, 24).transpose(3, 2, 0, 1)

        image_clean = torch.from_numpy(image_clean)
        image_noise = torch.from_numpy(image_noise)

        return image_clean, image_noise
    



def get_dataloaders(train_path_list, val_path_list, batch_size=1, drop_last=True, concat=True, n_worker=0):

    batch_sizes = {'train': batch_size, 'val': 1}
    tfs = []
    tfs += [
    MyToTensor()
    ]
    train_transforms = transforms.Compose(tfs)
    test_transforms = transforms.Compose([MyToTensor()])
    data_transforms = {'train': train_transforms, 'val': test_transforms}
    if concat:
        train = torch.utils.data.ConcatDataset(
            [Dataset(train_path_list, data_transforms['train']) for _ in range(batch_sizes['train'])])
    else:
        train = Dataset(train_path_list, data_transforms['train'])

    image_datasets = {'train': train,
                      'val': Dataset(val_path_list, data_transforms['val'])}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                  num_workers=n_worker,drop_last=drop_last, shuffle=(x == 'train')) for x in ['train',  'val']}
    return dataloaders

def flipit(image, axes):

    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)

    return image
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def collate_wrapper(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_wrapper([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

