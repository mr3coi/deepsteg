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
    def __init__(self, path):
        '''
        :param path: path to the directory containing the images
        :type path: str
        '''
        self._covers_path = os.path.join(path, 'covers')
        self._secrets_path = os.path.join(path, 'secrets')
        self._covers = os.listdir(self._covers_path)
        self._secrets = os.listdir(self._secrets_path)

    def __len__(self):
        return len(self._covers)

    def __getitem__(self, idx):
        cover_image = Image.open(os.path.join(self._covers_path, self._covers[idx])).convert('RGB')
        secret_image = Image.open(os.path.join(self._secrets_path, self._secrets[idx])).convert('RGB')
        arrays = [np.array(image).astype(float).transpose(2,0,1) / 256 for image in (cover_image, secret_image)]
        arrays = [torch.from_numpy(arr).float() for arr in arrays]    # convert to 'torch.Tensor'
        return arrays

def get_loaders(root_path: str, batch_size: int, shuffle=True, num_workers=2):
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
    subsets = ['train','val','test'] 
    paths = [os.path.join(root_path, subset) for subset in subsets]
    loaders = [DataLoader(dataset = ImageDataset(path),
                          batch_size = batch_size,
                          shuffle=shuffle,
                          num_workers = num_workers) for path in paths]
    return dict(zip(subsets, loaders))

