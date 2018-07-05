import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce

# TODO also return IRs

'''
Notes:
    - All image inputs are assumed to be in RGB format
    - 'pnet'(p) : prep network, 'hnet'(h) : hide network, 'rnet'(r) : reveal network
        (refer to the original paper for their corresponding meanings)
    - Output image is in 0-1 float format
'''
class DeepSteg(nn.Module):
    def __init__(self, batch_size, im_dim, c="RGB", s="RGB"):
        '''
        :param im_dim: the dimensions(height,width) of input image
        :type im_dim: tuple(int,int)
        '''
        super(DeepSteg, self).__init__()

        # Member variables
        self._image_dim = im_dim
        self._batch_size = batch_size
        channels = {"RGB":3,"L":1}

        # Prep network - TODO implement
        self._prep_out_channels = channels[s]

        # Hide network
        self._h_layer_num = 5
        self._h_kernel_type = [1,3,5]
        self._h_kernel_num = len(self._h_kernel_type)

        self._h_in_channels = [channels[c] + self._prep_out_channels,] \
                + [50 * self._h_kernel_num,] * (self._h_layer_num - 1)     # input dim
        self._h_out_channels = [50 * self._h_kernel_num,] * (self._h_layer_num - 1) + [channels[c],]     # final output dim
        self._h_kernel_sizes = [[val,] * self._h_layer_num for val in self._h_kernel_type]
        self._h_paddings = [[layer_kernel // 2 for layer_kernel in kernel_size] \
                                for kernel_size in self._h_kernel_sizes]

        hnets = [[nn.Conv2d(c_in, c_out // self._h_kernel_num, kernel_size=kernel, stride=1, padding=pad) \
                        for c_in, c_out, kernel, pad \
                        in zip(self._h_in_channels, self._h_out_channels, kernel_size, paddings)] \
                    for kernel_size, paddings \
                    in zip(self._h_kernel_sizes, self._h_paddings)]
        h_batchnorms = [nn.BatchNorm2d(num_features = C) \
                for C in self._h_out_channels]          # Batch Norm. (input channel: concatenated)
        h_nls = [nn.ReLU(),] * (self._h_layer_num-1) + [nn.Sigmoid(),]      # Non-linearities

        #self._hnet = nn.ModuleList(list(reduce(lambda a,b:a+b, zip(hnet, h_batchnorms, h_nls))))
        self._hnet = nn.ModuleList(nn.ModuleList(layers_list) for layers_list in (*hnets, h_batchnorms, h_nls))

        # Reveal network
        self._r_layer_num = 5
        self._r_in_channels = [channels[c],] + [50] * (self._r_layer_num - 1)     # input dim
        self._r_out_channels = [50] * (self._r_layer_num - 1) + [channels[s],]
        self._r_kernel_sizes = [5,5,5,5,5]
        self._r_paddings = [kernel // 2 for kernel in self._r_kernel_sizes]
        rnet = [nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=pad) \
                for c_in, c_out, kernel, pad \
                in zip(self._r_in_channels, self._r_out_channels, \
                    self._r_kernel_sizes, self._r_paddings)]
        r_batchnorms = [nn.BatchNorm2d(num_features=C) for C in self._r_out_channels]   # Batch Norm.
        r_nls = [nn.ReLU(),] * (self._h_layer_num-1) + [nn.Sigmoid(),]      # Non-linearities

        self._rnet = nn.ModuleList(list(reduce(lambda a,b:a+b, zip(rnet, r_batchnorms, r_nls))))

    def forward(self, covers, secrets, device, noise_level=None):
        # Run prep network - TODO implement => out = !
        #images['prepped'] = torch.Tensor(out)
        prep_out = secrets   # TODO replace w/ filtered version once implemented

        # Run hiding network
            # concatenate in the last dimension (channels)
        hidden_out = torch.cat((covers,prep_out), dim=1).to(device)    # hidden_in

        '''
        for layer in self._hnet:
            hidden_out = layer(hidden_out)
        '''

        for conv1, conv2, conv3, batchnorm, nls in zip(*self._hnet):
            conv_res1 = conv1(hidden_out)
            conv_res2 = conv2(hidden_out)
            conv_res3 = conv3(hidden_out)
            conv_res = [conv_res1, conv_res2, conv_res3]
            hidden_out = torch.cat(conv_res, dim=1)
            hidden_out = batchnorm(hidden_out)
            hidden_out = nls(hidden_out)

        # Run reveal network
        revealed_out = hidden_out.clone().to(device)
        if noise_level > 0:
            noise = revealed_out.new_empty(size=revealed_out.size(),requires_grad=False).normal_()
            noise = noise.mul(noise_level)
            revealed_out = F.sigmoid(revealed_out.add(noise))
        for idx, layer in enumerate(self._rnet):
            revealed_out = layer(revealed_out)

        return prep_out, hidden_out, revealed_out

    @property
    def image_dim(self):
        return self._image_dim

    @property
    def batch_size(self):
        return self._batch_size

