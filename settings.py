import argparse
parser = argparse.ArgumentParser()

# optional argument
parser.add_argument('--LAMBDA',default=10,type=int,help="Gradient penalty lambda hyperparameter")
parser.add_argument('--epochs',default=30,type=int,help="How many generator epochs to train for")
parser.add_argument('--critic_iters',default=5,type=int,help="For WGAN and WGAN-GP, number of critic iters per gen iter")
parser.add_argument('--output_dim', default=784, type=int,help="Number of pixels in MNIST (28*28)")
parser.add_argument('--use_cuda',default=True,type=bool,help="whether use CUDA")
parser.add_argument('--gpu',default=0,type=int,help="if use CUDA, which gpu to use")
# ################### stage 1 skeleton ###################
# hparams: in_channels,out_channels,latent_dim,hidden_dims,extract_layers,input_size,cond_size, kld_loss_weight,
#                  cond_f_dims=128,block=BasicBlock
# param overwrites
# parser.set_defaults(gradient_clip_val=5.0)
# TODO change params now we use black-white map
# network params
# namely, input num of key points
parser.add_argument('--in_channels', default=1, type=int)
# namely, output num of key points
parser.add_argument('--out_channels', default=1, type=int)
# use 500 for CPU, 50000 for GPU to see speed difference
parser.add_argument('--hidden_dims', default=0, type=int)  # default [32,64,128,256,512]
parser.add_argument('--latent_dim', default=128, type=int)  # 128 or 256 not sure
parser.add_argument('--extract_layers', default=0, type=int)  # default [2,2,2,2]
# input_size of heatmap or skeleton
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--cond_size', default=256, type=int)  # condition img size
# not sure how to set weights between KLd and recon loss
parser.add_argument('--kld_loss_weight', default=0.2, type=float)
# dims of feature extractor for cond images
parser.add_argument('--cond_f_dims', default=128, type=int)
parser.add_argument('--dataset_vae', default="D:/download_cache/VAEmodel", type=str, help="upper folder for train and val data")
parser.add_argument('--transformer_flag', default=True, type=bool, help="whether to use transformer")

# training params (opt)
parser.add_argument('--optimizer_name', default='adam', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr_vae', default=0.005, type=float)
# ################### stage 2 content ###################

args = parser.parse_args()
# print(args.sample_info)
