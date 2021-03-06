import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce

'''
Notes:
    - All image inputs are assumed to be in RGB format
    - 'pnet'(p) : prep network, 'hnet'(h) : hide network, 'rnet'(r) : reveal network
        (refer to the original paper for their corresponding meanings)
    - Output image is in 0-1 float format
'''
class DeepSteg(nn.Module):
    def __init__(self, batch_size, im_dim, c="RGB", s="RGB", skip=False):
        '''
        :param im_dim: the dimensions(height,width) of input image
        :type im_dim: tuple(int,int)
        '''
        super(DeepSteg, self).__init__()

        # Member variables
        self._image_dim = im_dim
        self._batch_size = batch_size
        channels = {"RGB":3,"L":1}
        self._h_skips = None

        # =========================== Prep network ===================================
        self._p_out_channel = 3     # assumed to be a multiple of '_p_kernel_num'

        ### Convolution layers
        self._p_layer_num = 5
        self._p_kernel_type = [1,3,5]
        self._p_kernel_num = len(self._p_kernel_type)
        self._p_kernel_sizes = [[val,] * self._p_layer_num for val in self._p_kernel_type]

        ### Concat version
        self._p_in_channels = [channels[s],] + [50 * self._p_kernel_num,] * (self._p_layer_num - 1)     # input dim
        self._p_out_channels = [50 * self._p_kernel_num,] * (self._p_layer_num - 1) + [self._p_out_channel,]     # final output dim
        p_nets = [[nn.Conv2d(c_in, c_out // self._p_kernel_num, kernel_size=kernel, stride=1, padding=(kernel//2)) \
                        for c_in, c_out, kernel \
                        in zip(self._p_in_channels, self._p_out_channels, kernel_size)] \
                    for kernel_size \
                    in self._p_kernel_sizes]
        ### Summation version
        '''
        self._p_in_channels = [channels[s],] + [50,] * (self._p_layer_num - 1)
        self._p_out_channels = [50,] * (self._p_layer_num - 1) + [self._p_out_channel,]
        p_nets = [[nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=(kernel//2)) \
                        for c_in, c_out, kernel \
                        in zip(self._p_in_channels, self._p_out_channels, kernel_size)] \
                    for kernel_size \
                    in self._p_kernel_sizes]
        '''

        ### Batch Norm. (input channel: concatenated)
        p_batchnorms = [nn.BatchNorm2d(num_features = C) for C in self._p_out_channels]

        ### Non-linearities
        p_nls = [nn.ReLU(),] * (self._p_layer_num-1) + [nn.Sigmoid(),]

        self._p_net = nn.ModuleList(nn.ModuleList(layers_list) \
                                        for layers_list \
                                        in (*p_nets, p_batchnorms, p_nls))

        # =========================== Hide network ===================================
        ### Convolution layers
        self._h_layer_num = 5
        self._h_kernel_type = [1,3,5]
        self._h_kernel_num = len(self._h_kernel_type)
        self._h_kernel_sizes = [[val,] * self._h_layer_num for val in self._h_kernel_type]

        ### Concat version
        self._h_in_channels = [channels[c] + self._p_out_channel,] \
                               + [50 * self._h_kernel_num,] * (self._h_layer_num - 1)     # input dim
        self._h_out_channels = [50 * self._h_kernel_num,] * (self._h_layer_num - 1) + [channels[c],]     # final output dim
        if skip:
            self._h_skips = [(0,2),(2,4)]       # (layer_in, layer_out) (in terms of # input channels)
            for skip_in, skip_out in self._h_skips:
                self._h_in_channels[skip_out] += self._h_in_channels[skip_in]

        h_nets = [[nn.Conv2d(c_in, c_out // self._h_kernel_num, kernel_size=kernel, stride=1, padding=(kernel//2)) \
                        for c_in, c_out, kernel \
                        in zip(self._h_in_channels, self._h_out_channels, kernel_size)] \
                    for kernel_size \
                    in self._h_kernel_sizes]
        ### Summation version
        '''
        self._h_in_channels = [channels[c] + self._p_out_channel,] + [50,] * (self._h_layer_num - 1)
        self._h_out_channels = [50,] * (self._h_layer_num - 1) + [channels[c],]
        h_nets = [[nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=(kernel//2) ) \
                        for c_in, c_out, kernel \
                        in zip(self._h_in_channels, self._h_out_channels, kernel_size)] \
                    for kernel_size \
                    in self._h_kernel_sizes]
        '''

        ### Batch Norm. (input channel: concatenated)
        h_batchnorms = [nn.BatchNorm2d(num_features = C) for C in self._h_out_channels]

        ### Non-linearities
        h_nls = [nn.ReLU(),] * (self._h_layer_num-1) + [nn.Sigmoid(),]

        self._h_net = nn.ModuleList(nn.ModuleList(layers_list) for layers_list in (*h_nets, h_batchnorms, h_nls))

        # =========================== Reveal network ===================================
        ### Convolution layers
        self._r_layer_num = 5
        self._r_kernel_type = [1,3,5]
        self._r_kernel_num = len(self._r_kernel_type)
        self._r_kernel_sizes = [[val,] * self._r_layer_num for val in self._r_kernel_type]

        ### Concat version
        self._r_in_channels = [channels[c],] + [50 * self._r_kernel_num,] * (self._r_layer_num - 1)     # input dim
        self._r_out_channels = [50 * self._r_kernel_num,] * (self._r_layer_num - 1) + [channels['RGB'],]     # final output dim
        r_nets = [[nn.Conv2d(c_in, c_out // self._r_kernel_num, kernel_size=kernel, stride=1, padding=(kernel//2)) \
                        for c_in, c_out, kernel \
                        in zip(self._r_in_channels, self._r_out_channels, kernel_size)] \
                    for kernel_size \
                    in self._r_kernel_sizes]
        ### Summation version
        '''
        self._r_in_channels = [channels[c],] + [50,] * (self._r_layer_num - 1)     # input dim
        self._r_out_channels = [50,] * (self._r_layer_num - 1) + [channels[s],]     # final output dim
        self._r_paddings = [[layer_kernel // 2 for layer_kernel in kernel_size] \
                                for kernel_size in self._r_kernel_sizes]

        r_nets = [[nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=(kernel//2)) \
                        for c_in, c_out, kernel \
                        in zip(self._r_in_channels, self._r_out_channels, kernel_size)] \
                    for kernel_size \
                    in self._r_kernel_sizes]
        '''

        ### Batch Norm. (input channel: concatenated)
        r_batchnorms = [nn.BatchNorm2d(num_features = C) for C in self._r_out_channels]

        ### Non-linearities
        r_nls = [nn.ReLU(),] * (self._r_layer_num-1) + [nn.Sigmoid(),]

        self._r_net = nn.ModuleList(nn.ModuleList(layers_list) \
                                        for layers_list \
                                        in (*r_nets, r_batchnorms, r_nls))

        ### Extra layer for grayscale output (if input secret image is in grayscale)
        if s == 'L':
            self._gray_net = nn.Conv2d(channels['RGB'], channels['L'], kernel_size=1, stride=1, padding=0)
        else:
            self._gray_net = None

    def forward(self, covers, secrets, device, noise_level=0):
        # Run prep network
        prep_out = secrets

        for conv1, conv2, conv3, batchnorm, nls in zip(*self._p_net):
            conv_res1 = conv1(prep_out)
            conv_res2 = conv2(prep_out)
            conv_res3 = conv3(prep_out)
            prep_out = torch.cat([conv_res1, conv_res2, conv_res3], dim=1)
            #prep_out = reduce(torch.add, [conv_res1, conv_res2, conv_res3])
            prep_out = batchnorm(prep_out)
            prep_out = nls(prep_out)

        # Run hiding network
        ### concatenate in the channels dimension
        '''
        hidden_out = torch.cat((covers,prep_out), dim=1).to(device)    # hidden_in

        for it, (conv1, conv2, conv3, batchnorm, nls) in enumerate(zip(*self._h_net)):
            conv_res1 = conv1(hidden_out)
            conv_res2 = conv2(hidden_out)
            conv_res3 = conv3(hidden_out)
            hidden_out = torch.cat([conv_res1, conv_res2, conv_res3], dim=1)
            #hidden_out = reduce(torch.add, [conv_res1, conv_res2, conv_res3])
            hidden_out = batchnorm(hidden_out)
            hidden_out = nls(hidden_out)
        '''
        layer_inout = []
        layer_inout.append(torch.cat((covers,prep_out), dim=1).to(device))    # hidden_in

        if self._h_skips is not None:
            do_skip = True
            skip_iter = iter(self._h_skips)
            skip_src, skip_dest = next(skip_iter, None)

        for it, (conv1, conv2, conv3, batchnorm, nls) in enumerate(zip(*self._h_net)):
            if do_skip and it == skip_dest:
                new_hidden_in = torch.cat([layer_inout[-1], layer_inout[skip_src]], dim=1)
                skip_items = next(skip_iter, None)
                if skip_items is None:
                    do_skip = False
                else:
                    skip_src, skip_dest = skip_items
                layer_inout[-1] = new_hidden_in
            conv_res1 = conv1(layer_inout[-1])
            conv_res2 = conv2(layer_inout[-1])
            conv_res3 = conv3(layer_inout[-1])
            hidden_out = torch.cat([conv_res1, conv_res2, conv_res3], dim=1)
            #hidden_out = reduce(torch.add, [conv_res1, conv_res2, conv_res3])
            hidden_out = batchnorm(hidden_out)
            hidden_out = nls(hidden_out)
            layer_inout.append(hidden_out)

        # Run reveal network
        revealed_out = hidden_out.clone().to(device)
        if noise_level > 0:
            noise = revealed_out.new_empty(size=revealed_out.size(),requires_grad=False).normal_()
            noise = noise.mul(noise_level)
            revealed_out = F.sigmoid(revealed_out.add(noise))

        for it, (conv1, conv2, conv3, batchnorm, nls) in enumerate(zip(*self._r_net)):
            conv_res1 = conv1(revealed_out)
            conv_res2 = conv2(revealed_out)
            conv_res3 = conv3(revealed_out)
            revealed_out = torch.cat([conv_res1, conv_res2, conv_res3], dim=1)
            #revealed_out = reduce(torch.add, [conv_res1, conv_res2, conv_res3])
            revealed_out = batchnorm(revealed_out)
            revealed_out = nls(revealed_out)

        if self._gray_net is not None:
            revealed_out = self._gray_net(revealed_out)

        return prep_out, hidden_out, revealed_out

    @property
    def image_dim(self):
        return self._image_dim

    @property
    def batch_size(self):
        return self._batch_size

