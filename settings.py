import argparse
parser = argparse.ArgumentParser()

parser.add_argument('data',  metavar='DIR', help='path to dataset')
# optional argument
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--dim',  default=64,type=int,help="input_dim")
parser.add_argument('--LAMBDA',  default=10,type=int,help="Gradient penalty lambda hyperparameter")
parser.add_argument('--iters',  default=200000,type=int,help="How many generator iterations to train for")
parser.add_argument('--critic_iters',  default=5,type=int,help="For WGAN and WGAN-GP, number of critic iters per gen iter")
parser.add_argument('--output_dim',  default=784,type=int,help="Number of pixels in MNIST (28*28)")
parser.add_argument('--use_cuda',  default=True,type=bool,help="whether use CUDA")
parser.add_argument('--gpu',  default=0,type=int,help="if use CUDA, which gpu to use")


args = parser.parse_args()
# print(args.sample_info)