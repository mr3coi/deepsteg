import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse
import os
import pickle

from deepsteg import DeepSteg
from data_loader import get_loaders # TODO add ImageNetDataset if needed

import visdom
import nsml
from nsml import DATASET_PATH, Visdom

def get_parser():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('-e', "--epochs", default=100,
                        help="Number of epochs to run training in.")
    parser.add_argument("--batch-size", default=25,
                        help="Size of a single minibatch.")
    parser.add_argument("--learning-rate", default=0.001,
                        help="Size of the learning rate for training.")
    parser.add_argument("--beta", default=1.0,
                        help="The scaling parameter described in the paper.")
    parser.add_argument("--c-mode", default="RGB",
                        help="The image mode for the cover images.")
    parser.add_argument("--s-mode", default="RGB",
                        help="The image mode for the secret images.")
    parser.add_argument("--noise-level", type=float, default=0.004,     # 1/256 = (approx.) 0.004
                        help="The scaling parameter for the noise added to the reveal network.")

    # PyTorch-related
    parser.add_argument("--no-cuda", action='store_true',
                        help="disable use of CUDA and nVIDIA GPUs.")
    parser.add_argument("--shuffle", dest='shuffle', action='store_true',
                        help="Shuffle the dataset when generating DataLoaders.")
    parser.add_argument("--no-shuffle", dest='shuffle', action='store_false',
                        help="Do not shuffle the dataset when generating DataLoaders.")

    # Others
    parser.add_argument('-d', "--draft", action='store_true',
                        help="Conduct a test run consisting of a single epoch.")
    parser.add_argument('-l', "--local", action='store_true',
                        help="Conduct a test run on the local machine (instead of NSML server).")
    parser.add_argument('-v', "--verbose", action='store_true',
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
    # Set mode to 'train' (for batch norm.)
    model.train()

    # Enable CUDA if specified
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Initialize visdom object
    if not args.local:
        viz = Visdom(visdom=visdom)

    # Specify loss function & optimizer
    criterion = nn.L1Loss()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params = parameters, lr = args.learning_rate)

    # Compute the number of iterations in an epoch
    num_iters = len(train_loader)
    if int(args.max_dataset_size) > 0:
        num_iters = int(args.max_dataset_size) // int(args.batch_size)

    min_loss = 10000000

    # For NSML saving function
    bind_model(model=model, optimizer=optimizer)

    # Conduct training
    for epoch in range(int(args.epochs)):
        total_loss = 0

        if args.verbose:
            print('=============== Epoch {} ==============='.format(epoch+1))

        for it, (covers, secrets) in enumerate(train_loader):
            print('\tIter: {} / {}'.format(it+1, num_iters))

            # Enable CUDA if specified
            covers, secrets = covers.to(device), secrets.to(device)

            # Forward run
            prepped, container, revealed = model.forward(covers, secrets, device=device,
                                                         noise_level=float(args.noise_level))

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
                print("\t\tLoss at iter {}: {}".format(it+1, loss.item()))

            if float(args.max_dataset_size) > 0 and (it+1) * int(args.batch_size) >= int(args.max_dataset_size):
                break

        if args.verbose:
            print('Loss at epoch {}: {}'.format(epoch+1, total_loss / num_iters))

        # Report statistics to NSML (if not draft)
        if not args.local:
            nsml.report(summary = True,
                        step = (epoch+1) * len(train_loader),
                        epoch_total = int(args.epochs),
                        train__loss = total_loss / num_iters)

            # Visualize input & output images
            images = [images.detach()[0] for images in [covers, secrets, container, revealed]]
            if use_cuda:
                images = [image.cpu() for image in images]
            images = [image.numpy() for image in images]
            if args.c_mode == args.s_mode:
                viz.images(images, opts=dict(nrow=1, title='epoch_{}'.format(epoch+1)))
            else:
                for title, image in zip(['cover','secret','container','revealed'],images):
                    viz.image(image, opts=dict(title=title,caption='epoch_{}'.format(epoch+1)))

        # Save session if minimum loss is renewed
        if not args.draft and not args.local and total_loss < min_loss:
            nsml.save(epoch)
            min_loss = total_loss

        if args.draft:
            break

def bind_model(model, optimizer=None, sample=None):
    def save(filename, **kwargs):
        '''
        :param filename: root directory for the saved model (subdirectories: 'model', 'sample')
        '''
        if not os.path.isdir(filename):
            os.makedirs(filename)

        state = dict(model=model.state_dict())
        if optimizer:
            state['optimizer'] = optimizer.state_dict()
        torch.save(state, os.path.join(filename, 'state.pt'))

        print(">>> State has been successfully saved, at {}".format(filename))

    def load(filename):
        state_path = os.path.join(filename, 'state.pt')
        state = torch.load(state_path)
        model.load_state_dict(state['model'])
        if optimizer and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        print(">>> State has been successfully loaded.")

    nsml.bind(save=save, load=load)

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup DataLoader
    if args.local:
        ROOT_PATH = os.path.join(args.local_dataset_path, 'train') # Root directory for the dataset
    else:
        ROOT_PATH = os.path.join(DATASET_PATH, 'train') # Root directory for the dataset
    loaders = get_loaders(root_path = ROOT_PATH,
                          batch_size = int(args.batch_size),
                          shuffle = args.shuffle,
                          c_mode = args.c_mode,
                          s_mode = args.s_mode)

    # Run training
    model = DeepSteg(batch_size = int(args.batch_size), im_dim = (255,255),
                     c=args.c_mode, s=args.s_mode)        # TODO fix when data is replaced

    train(model = model,
          train_loader = loaders['train'],
          args = args,
          beta = float(args.beta))
    print("all successfully completed")

if __name__ == "__main__":
    main()
