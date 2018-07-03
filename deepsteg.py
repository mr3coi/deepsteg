import torch
from torch import nn
from functools import reduce
from copy import deepcopy

'''
Notes:
    - All image inputs are assumed to be in RGB format
    - 'pnet'(p) : prep network, 'hnet'(h) : hide network, 'rnet'(r) : reveal network
        (refer to the original paper for their corresponding meanings)
    - Output image is in 0-1 float format
'''
class DeepSteg(nn.Module):
    def __init__(self, batch_size, im_dim):
        '''
        :param im_dim: the dimensions(height,width) of input image
        :type im_dim: tuple(int,int)
        '''
        super(DeepSteg, self).__init__()

        # Member variables
        self._image_dim = im_dim
        self._batch_size = batch_size
        self._image_channels = 3    # Assuming RGB input / output

        # Prep network - TODO implement
        self._prep_out_channels = 3

        # Hide network
        self._h_layer_num = 5
        self._h_in_channels = [self._image_channels + self._prep_out_channels,] \
                + [50] * (self._h_layer_num - 1)     # input dim
        self._h_out_channels = [50] * (self._h_layer_num - 1) + [self._image_channels,]     # final output dim
        self._h_kernel_sizes = [5,5,5,5,5]
        self._h_paddings = [kernel // 2 for kernel in self._h_kernel_sizes]
        hnet = [nn.Conv2d(c_in, c_out, kernel, pad) \
                    for c_in, c_out, kernel, pad \
                    in zip(self._h_in_channels, self._h_out_channels, \
                        self._h_kernel_sizes, self._h_paddings)]
        h_nls = [nn.ReLU()] * 4 + [nn.Sigmoid()]    # Non-linearities

        self._hnet = reduce(lambda a,b:a+b, zip(hnet, h_nls))

        # Reveal network
        self._r_layer_num = 5
        self._r_in_channels = [self._image_channels + self._prep_out_channels,] \
                + [50] * (self._r_layer_num - 1)     # input dim
        self._r_out_channels = [50] * (self._r_layer_num - 1) + [self._image_channels,]     # final output dim
        self._r_kernel_sizes = [5,5,5,5,5]
        self._r_paddings = [kernel // 2 for kernel in self._r_kernel_sizes]
        rnet = [nn.Conv2d(c_in, c_out, kernel, pad) \
                for c_in, c_out, kernel, pad \
                in zip(self._r_in_channels, self._r_out_channels, \
                    self._r_kernel_sizes, self._r_paddings)]
        r_nls = [nn.ReLU()] * 4 + [nn.Sigmoid()]

        self._rnet = reduce(lambda a,b:a+b, zip(rnet, r_nls))

    def forward(self, covers, secrets):
        images = dict()

        # Run prep network - TODO implement => out = !
        #images['prepped'] = deepcopy(out)
        out = secrets   # TODO replace w/ filtered version once implemented

        # Run hiding network
            # concatenate in the last dimension (channels)
        out = torch.cat((covers,out), dim=1)
        for layer in self._hnet:
            out = layer.forward(out)

        images['container'] = deepcopy(out)

        # Run reveal network
        for layer in self._rnet:
            out = layer.forward(out)

        images['revealed'] = out

        return images

    @property
    def image_dim(self):
        return self._image_dim

    @property
    def batch_size(self):
        return self._batch_size

