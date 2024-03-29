from settings import args
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(1)
if ~torch.cuda.is_available():
    args.use_cuda = False
use_cuda = args.use_cuda
# import tflib as lib
# import tflib.save_images
# import tflib.mnist
# import tflib.plot
# def generate_image(frame, netG):
#     lib.print_model_settings(locals().copy())
#     noise = torch.randn(args.batch_size, 128)
#     if use_cuda:
#         noise = noise.cuda(args.gpu)
#     noisev = autograd.Variable(noise, volatile=True)
#     samples = netG(noisev)
#     samples = samples.view(args.batch_size, 28, 28)
#     # print samples.size()
#
#     samples = samples.cpu().data.numpy()
#
#     lib.save_images.save_images(
#         samples,
#         'tmp/mnist/samples_{}.png'.format(frame)
#     )



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



