import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse
import os

from deepsteg import DeepSteg
from data_loader import get_loaders # TODO add ImageNetDataset if needed

import nsml
from nsml import DATASET_PATH

def get_parser():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", default=10,
                        help="Number of epochs to run training in.")
    parser.add_argument("--batch-size", default=50,
                        help="Size of a single minibatch.")
    parser.add_argument("--learning-rate", default=0.001,
                        help="Size of the learning rate for training.")

    # PyTorch-related
    parser.add_argument("--no-cuda", action='store_true',
                        help="disable use of CUDA and nVIDIA GPUs.")
    parser.add_argument("--shuffle", dest='shuffle', action='store_true',
                        help="Shuffle the dataset when generating DataLoaders.")
    parser.add_argument("--no-shuffle", dest='shuffle', action='store_false',
                        help="Do not shuffle the dataset when generating DataLoaders.")

    # Others
    parser.add_argument("--draft", action='store_true',
                        help="Conduct a test run consisting of a single epoch.")
    return parser


def train(model, train_loader, args, beta=1, cuda=True):
    '''
    :param beta: weight of errors
    :type beta: float (>=0)
    '''
    # Enable CUDA if specified
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Specify loss function & optimizer
    criterion = nn.L1Loss()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params = parameters, lr = args.learning_rate)

    # Conduct training
    for epoch in range(args.epochs):
        total_loss = 0

        for it, (covers, secrets) in enumerate(train_loader):
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

            # Collect statistics
            total_loss += loss.item()

            if args.draft:
                print("Loss at iter {}: {}".format(it, loss.item))

        # Report statistics to NSML (if not draft)
        if not args.draft:
            nsml.report(summary = True,
                        epoch = epoch,
                        epoch_total = args.epochs,
                        train__loss = total_loss)

        if args.draft:
            break

def main():
    parser = get_parser()
    args = parser.parse_args()
    DATASET_PATH = './data/deepsteg_imagenet'

    # Setup DataLoader
    ROOT_PATH = os.path.join(DATASET_PATH, 'train') # Root directory for the dataset
    loaders = get_loaders(root_path = ROOT_PATH,
                          batch_size = args.batch_size,
                          shuffle = args.shuffle)

    # Run training
    model = DeepSteg(batch_size = args.batch_size, im_dim = (255,255))        # TODO fix when data is replaced

    train(model = model,
          train_loader = loaders['train'],
          args = args)
    print("all successfully completed")

if __name__ == "__main__":
    main()
