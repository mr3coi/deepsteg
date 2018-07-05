from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
import os

class ImageDataset(Dataset):
    '''
    Loads all Image images located in a single directory,
        whether it be training/validation/testing set.
    Assumed directory tree structure:
        'cover', 'secret' directories under root (designated by 'path')
    Transform the images into (C,H,W) format (from (H,W,C)), and
        into arrays of pixel values in float of range 0-1
    '''
    def __init__(self, path, c_mode='RGB', s_mode='RGB'):
        '''
        :param path: path to the directory containing the images
        :type path: str
        '''
        self._covers_path = os.path.join(path, 'covers')
        self._secrets_path = os.path.join(path, 'secrets')
        self._covers = os.listdir(self._covers_path)
        self._secrets = os.listdir(self._secrets_path)
        self._c_mode = c_mode
        self._s_mode = s_mode

    def __len__(self):
        return len(self._covers)

    def __getitem__(self, idx):
        cover_image = Image.open(os.path.join(self._covers_path, self._covers[idx])).convert(self._c_mode)
        secret_image = Image.open(os.path.join(self._secrets_path, self._secrets[idx])).convert(self._s_mode)
        arrays = []
        for image, mode in zip((cover_image, secret_image), (self._c_mode, self._s_mode)):
            arr = np.array(image) / 256
            if mode == 'L':
                arr = np.reshape(arr, arr.shape + (1,))
            arrays.append(arr)
        arrays = [arr.transpose(2,0,1) for arr in arrays]
        arrays = [torch.from_numpy(arr).float() for arr in arrays]    # convert to 'torch.Tensor'
        return arrays

def get_loaders(root_path: str, batch_size: int, shuffle=True, num_workers=2,
                c_mode='RGB', s_mode='RGB'):
    '''
    Create DataLoaders from the given database.
    Assumes that images are located in 'train', 'val', 'test' directories
        all located under the given root path.
    All loaders share the same minibatch size.

    :param root_path: path to the root directory containing subdirectories
                        'train', 'val', 'test'
    :type root_path: str
    :param batch_size: size of a minibatch for all loaders
    :type batch_size: int
    :param shuffle: 
    :type shuffle: bool
    :param num_workers: 
    :type num_workers: int 

    :return: a dictionary mapping 'train/test/val' to its corresponding loader
    :rtype: dictionary(str, torch.utils.data.DataLoader)
    '''
    assert type(batch_size) == int, "batch_size: required type 'int', but got {}".format(type(batch_size))
    subsets = ['train','val','test'] 
    paths = [os.path.join(root_path, subset) for subset in subsets]
    loaders = [DataLoader(dataset = ImageDataset(path, c_mode=c_mode, s_mode=s_mode),
                          batch_size = batch_size,
                          shuffle=shuffle,
                          num_workers = num_workers) for path in paths]
    return dict(zip(subsets, loaders))

