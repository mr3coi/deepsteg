from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy
import os

class ImageNetDataset(Dataset):
    '''
    Loads all ImageNet images located in a single directory,
        whether it be training/validation/testing set.
    Transform the images into (C,H,W) format (from (H,W,C)), and
        into arrays of pixel values in float of range 0-1
    '''
    def __init__(self, path):
        '''
        :param path: path to the directory containing the images
        :type path: str
        '''
        self._path = path
        self._items = os.listdir(path)

    def __len__(self):
        return len(self._items)

    def __getItem__(self, idx):
        image = Image.open(os.path.join(self._path, self._items[idx]))
        transformed = np.transpose(np.array(image), (2,0,1)).astype(float) / 256
        return transformed

def get_loaders(root_path: str, batch_size: int, shuffle=True, num_workers=2):
    '''
    Create DataLoaders from the given ImageNet database.
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
    loaders = [DataLoader(dataset = ImageNetDataset(path),
                          batch_size = batch_size,
                          shuffle=shuffle,
                          num_workers = num_workers) for path in paths]
    return dict(zip(subsets, loaders))

