import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse
import os

from deepsteg import DeepSteg
from data_loader import get_loaders # TODO add ImageNetDataset if needed

import nsml
from nsml import DATASET_PATH

# TODO check that inputs are in NCHW format => modify if not so

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10,
                        help="Number of epochs to run training in.")
    parser.add_argument("--batch-size", default=50,
                        help="Size of a single minibatch.")
    parser.add_argument("--learning-rate", default=0.001,
                        help="Size of the learning rate for training.")
    parser.add_argument("--no-cuda", action='store_true',
                        help="disable use of CUDA and nVIDIA GPUs.")
    parser.add_argument("--draft", action='store_true',
                        help="Conduct a test run consisting of a single epoch.")
    return parser

def train(model, beta=1, cuda=True):
    '''
    :param beta: weight of errors
    :type beta: float (>=0)
    '''
    # Enable CUDA if specified
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Specify loss function & optimizer
    criterion = nn.L1Loss()
    parameters= [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params = parameters,       # TODO other hyperparameters
                                 lr = args.learning_rate)

    # Setup DataLoader
    loaders = get_loaders()

    # Conduct training
    for epoch in range(args.epochs):
        for covers, secrets in minibatch:   # TODO fix
            # Enable CUDA if specified
            covers, secrets = covers.to(device), secrets.to(device)

            # Forward run
            outputs = model.forward(covers, secrets)

            # Compute loss
            loss = criterion(outputs['container'], covers)
            loss += beta * criterion(outputs['revealed'], secrets)

            # Do back-propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if args.draft:
            break

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Root directory for the dataset
    ROOT_PATH = os.path.join(DATASET_PATH, 'train')

    # CUDA setup
    use_cuda= not args.no_cuda and torch.cuda.is_available()

    model = DeepSteg(batch_size = args.batch_size,
                     im_dim = (128,128))

if __name__ == "__main__":
    main()
