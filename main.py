import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse
import os

from deepsteg import DeepSteg
from data_loader import get_loaders # TODO add ImageNetDataset if needed

import visdom
import nsml
from nsml import DATASET_PATH, Visdom

def get_parser():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", default=100,
                        help="Number of epochs to run training in.")
    parser.add_argument("--batch-size", default=25,
                        help="Size of a single minibatch.")
    parser.add_argument("--learning-rate", default=0.001,
                        help="Size of the learning rate for training.")
    parser.add_argument("--beta", default=1.0,
                        help="The scaling parameter described in the paper.")

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
    parser.add_argument("--local", action='store_true',
                        help="Conduct a test run on the local machine (instead of NSML server).")
    parser.add_argument("--verbose", action='store_true',
                        help="Print progress to log.")
    parser.add_argument("--local-dataset-path", default='./data/deepsteg_imagenet',
                        help="Conduct a test run on the local machine (instead of NSML server).")
    parser.add_argument("--max-dataset-size", default=0,
                        help="Conduct a test run on the local machine (instead of NSML server).")
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

    # Initialize visdom object
    viz = Visdom(visdom=visdom)

    # Specify loss function & optimizer
    criterion = nn.L1Loss()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params = parameters, lr = args.learning_rate)

    # Compute the number of iterations in an epoch
    num_iters = len(train_loader)

    # Conduct training
    for epoch in range(args.epochs):
        total_loss = 0

        if args.verbose:
            print('=============== Epoch {} ==============='.format(epoch+1))

        for it, (covers, secrets) in enumerate(train_loader):
            if float(args.max_dataset_size) > 0 and it * int(args.batch_size) >= int(args.max_dataset_size):
                num_iters = it
                break

            # Enable CUDA if specified
            covers, secrets = covers.to(device), secrets.to(device)

            # Forward run
            prepped, container, revealed = model.forward(covers, secrets, device=device)

            # Compute loss
            loss = criterion(container, covers)
            loss += beta * criterion(revealed, secrets)

            # Do back-propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect statistics
            total_loss += loss.item()

            if args.verbose:
                print("\tLoss at iter {}: {}".format(it+1, loss.item()))

        if args.verbose:
            print('Loss at epoch {}: {}'.format(epoch+1, total_loss / num_iters))

        # Report statistics to NSML (if not draft)
        if not args.local:
            nsml.report(summary = True,
                        step = epoch * len(train_loader),
                        epoch_total = args.epochs,
                        train__loss = total_loss / num_iters)

            # Visualize input & output images
            images = [images.detach()[0] for images in [covers, secrets, container, revealed]]
            if use_cuda:
                images = [image.cpu() for image in images]
            images = [image.numpy() for image in images]
            '''
            titles = ['Cover Image', 'Secret Image', 'Container Image', 'Revealed Image']
            titles = ['epoch_{}_'.format(epoch) + txt for txt in titles]
            images = dict(zip(titles, images))
            for title, image in images.items():
                viz.image(image, opts=dict(title=title))
            '''
            viz.images(images, opts=dict(nrow=1, title='epoch_{}'.format(epoch)))

        if args.draft:
            break

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup DataLoader
    if args.local:
        ROOT_PATH = os.path.join(args.local_dataset_path, 'train') # Root directory for the dataset
    else:
        ROOT_PATH = os.path.join(DATASET_PATH, 'train') # Root directory for the dataset
    loaders = get_loaders(root_path = ROOT_PATH,
                          batch_size = args.batch_size,
                          shuffle = args.shuffle)

    # Run training
    model = DeepSteg(batch_size = int(args.batch_size), im_dim = (255,255))        # TODO fix when data is replaced

    train(model = model,
          train_loader = loaders['train'],
          args = args,
          beta = float(args.beta))
    print("all successfully completed")

if __name__ == "__main__":
    main()
