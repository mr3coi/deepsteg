import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as tf

import argparse
import os
import pickle

from deepsteg import DeepSteg
from data_loader import get_loaders

import visdom
import nsml
from nsml import DATASET_PATH, Visdom
from itertools import product

def get_parser():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('-e', "--epochs", default=100, type=int,
                        help="Number of epochs to run training in.")
    parser.add_argument("--batch-size", default=25, type=int,
                        help="Size of a single minibatch.")
    parser.add_argument("--learning-rate", default=0.001, type=float,
                        help="Size of the learning rate for training.")
    parser.add_argument("--beta", default=1.0, type=float,
                        help="The scaling parameter described in the paper.")
    parser.add_argument("--noise-level", type=float, default=0.004,     # 1/256 = (approx.) 0.004
                        help="The scaling parameter for the noise added to the reveal network.")
    parser.add_argument("--loss", default='MSE',
                        help="The loss function to use for training (supported: 'L1', 'MSE').")
    parser.add_argument("--skip", action='store_true',
                        help="Use U-net style skipped connections (in the hide network).")
    parser.add_argument('-g', "--gamma", type=float, default=0.1,
                        help="Hyperparameter for weighting the perceptual loss (from relu2-2 of VGG16) \
                                added to the existing loss (set to 0 to disable).")

    # PyTorch-related
    parser.add_argument("--no-cuda", action='store_true',
                        help="disable use of CUDA and nVIDIA GPUs.")
    parser.add_argument("--shuffle", dest='shuffle', action='store_true',
                        help="Shuffle the dataset when generating DataLoaders.")
    parser.add_argument("--no-shuffle", dest='shuffle', action='store_false',
                        help="Do not shuffle the dataset when generating DataLoaders.")

    # Visualization
    parser.add_argument('-v', "--verbose", action='store_true',
                        help="Print progress to log.")
    parser.add_argument("--viz-count", default=1, type=int,
                        help="Number of samples to visualize \
                             (starting from the first image of the last minibatch).")
    parser.add_argument("--c-mode", default="RGB",
                        help="The image mode for the cover images (supported: 'RGB', 'L').")
    parser.add_argument("--s-mode", default="RGB",
                        help="The image mode for the secret images (supported: 'RGB' only).")
    parser.add_argument("--c-draw-rgb", action='store_true',
                        help="Show & compare intensities of each channel of the cover and container images. \
                        Ignored if --c-mode == 'L'.")
    parser.add_argument("--s-draw-rgb", action='store_true',
                        help="Show & compare intensities of each channel of the secret and revealed images. \
                        Ignored if --s-mode == 'L'.")

    # Others
    parser.add_argument('-d', "--draft", action='store_true',
                        help="Conduct a test run consisting of a single epoch.")
    parser.add_argument('-l', "--local", action='store_true',
                        help="Conduct a test run on the local machine (instead of NSML server).")
    parser.add_argument("--local-dataset-path", default='./data/deepsteg_imagenet',
                        help="Conduct a test run on the local machine (instead of NSML server).")
    parser.add_argument("--max-dataset-size", default=0, type=int,
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
    if args.loss == 'L1':
        criterion = nn.L1Loss()
    elif args.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Only L1Loss(cmd arg:'L1') and MSELoss(cmd arg:'MSE') are supported.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params = parameters, lr = args.learning_rate)

    # Compute the number of iterations in an epoch
    num_iters = len(train_loader)
    if args.max_dataset_size > 0:
        num_iters = args.max_dataset_size // args.batch_size

    min_loss = 10000000

    # For NSML saving function
    bind_model(model=model, optimizer=optimizer)

    # Add perceptual loss term (use upto 'relu2-2' layer of VGG-16)
    if args.gamma > 0:
        vgg16_model = models.vgg16_bn(pretrained=True).to(device)
        layers_list = list(vgg16_model.features.children())[:14]
        '''
        for idx in [2,5,9,12]:
            layers_list[idx] = nn.ReLU(inplace=False)
        for layer in layers_list:
            print(layer)
        '''
        perceptual_model = torch.nn.Sequential(*layers_list).to(device)
        for param in perceptual_model.parameters():
            param.requires_grad = False
        
        perceptual_criterion = nn.MSELoss()

    # Conduct training
    for epoch in range(args.epochs):
        total_loss = 0

        if args.verbose:
            print('=============== Epoch {} ==============='.format(epoch+1))

        for it, (covers, secrets) in enumerate(train_loader):
            print('\tIter: {} / {}'.format(it+1, num_iters))

            # Enable CUDA if specified
            covers, secrets = covers.to(device), secrets.to(device)

            # Forward run
            prepped, container, revealed = model.forward(covers, secrets, device=device,
                                                         noise_level=args.noise_level)

            # Compute loss
            loss = criterion(container, covers)
            loss += beta * criterion(revealed, secrets)

            if args.gamma > 0:
                # Normalize each image in a minibatch of {covers, container}
                size = container.size()
                means = torch.tensor([0.485, 0.456, 0.406])
                stds = torch.tensor([0.229, 0.224, 0.225])
                means, stds = [item.view(3,1,1).expand(container.size()).to(device) for item in [means, stds]]
                normalized = [torch.div(torch.add(minibatch,-1,means),stds).to(device) \
                                for minibatch in [container, covers]]
                loss += args.gamma * criterion(*normalized)

            # Do back-propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect statistics
            total_loss += loss.item()

            if args.verbose:
                print("\t\tLoss at iter {}: {}".format(it+1, loss.item()))

            if args.max_dataset_size > 0 and (it+1) * args.batch_size >= args.max_dataset_size:
                break

        if args.verbose:
            print('Loss at epoch {}: {}'.format(epoch+1, total_loss / num_iters))

        # Report statistics to NSML (if not draft)
        if not args.local:
            nsml.report(summary = True,
                        step = (epoch+1) * len(train_loader),
                        epoch_total = args.epochs,
                        train__loss = total_loss / num_iters)

            # Visualize input & output images
            for idx in range(args.viz_count):
                images = [images.detach()[idx] \
                            for images in [covers, secrets, container, revealed]]   # select example images
                visualize(viz, images, epoch,
                          c_mode=args.c_mode, s_mode = args.s_mode,
                          c_draw_rgb=args.c_draw_rgb, s_draw_rgb = args.s_draw_rgb,
                          use_cuda = use_cuda)

        # Save session if minimum loss is renewed
        if not args.draft and not args.local and total_loss < min_loss:
            nsml.save(epoch)
            min_loss = total_loss

        if args.draft:
            break

def visualize(viz, images, epoch, c_mode='RGB', s_mode='RGB',
              c_draw_rgb=False, s_draw_rgb=False, use_cuda=False):
    '''
    :param images: list of [covers, secrets, container, revealed] images
    :type images: list(torch.Tensor) or list(torch.cuda.Tensor)
    '''
    # Copy images to host memory
    if use_cuda:
        images = [image.cpu() for image in images]

    # Conver to numpy arrays
    images = [image.numpy() for image in images]

    # Visualize the images
    if c_mode == s_mode:      # use container if applicable
        viz.images(images, opts=dict(nrow=1, title='epoch_{}'.format(epoch+1)))
    else:                               # render separately
        for title, image in zip(['cover','secret','container','revealed'],images):
            viz.image(image, opts=dict(title=title,caption='epoch_{}'.format(epoch+1)))

    # Visualize RGB channels separately
    cover, secret, container, revealed = images
    if c_mode == 'RGB' and c_draw_rgb:
        targets = [cover, container]
        viz.images([image[channel].reshape((1,) + image.shape[1:]) \
                        for channel, image in product(list(range(3)),targets)],
                   opts=dict(title="RGB comparison : epoch {}".format(epoch),
                             caption="R_cover, R_container, G_cover, G_container, B_cover, B_container"))
    if s_mode == 'RGB' and s_draw_rgb:
        targets = [secret, revealed]
        viz.images([image[channel].reshape((1,) + image.shape[1:]) \
                        for channel, image in product(list(range(3)),targets)],
                   opts=dict(title="RGB comparison : epoch {}".format(epoch),
                             caption="R_secret, R_revealed, G_secret, G_revealed, B_secret, B_revealed"))

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
                          batch_size = args.batch_size,
                          shuffle = args.shuffle,
                          c_mode = args.c_mode,
                          s_mode = args.s_mode)

    # Run training
    model = DeepSteg(batch_size = args.batch_size, im_dim = (255,255),
                     c=args.c_mode, s=args.s_mode, skip=args.skip)        # TODO fix when data is replaced

    train(model = model,
          train_loader = loaders['train'],
          args = args,
          beta = args.beta)
    print("all successfully completed")

if __name__ == "__main__":
    main()
