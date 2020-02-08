from settings import args
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from model import Generator,Discriminator
import time
import numpy as np

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

torch.manual_seed(1)
if ~torch.cuda.is_available():
    args.use_cuda = False

use_cuda = args.use_cuda


# test components TODO fit the modified model
def generate_image(frame, netG):
    #TODO change model G pipeline.
    lib.print_model_settings(locals().copy())
    noise = torch.randn(args.batch_size, 128)
    if use_cuda:
        noise = noise.cuda(args.gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(args.batch_size, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        'tmp/mnist/samples_{}.png'.format(frame)
    )



# WGAN special part.
def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(args.gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(args.gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(args.gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.LAMBDA
    return gradient_penalty


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)

class BasicBlock(nn.Module):
    expansion = 1
    #inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

