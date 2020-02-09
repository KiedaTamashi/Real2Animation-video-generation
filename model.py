from dataset import SkeletonTrainDataset, SkeletonValDataset
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from argparse import ArgumentParser
import logging as log
import pytorch_lightning as ptl
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from components import *
from settings import args
import os
import sys
sys.path.append(os.getcwd())


# ==================Definition Start - Stage 1======================


class skeletonVAE(ptl.LightningModule):
    def __init__(self, block=BasicBlock):
        super(skeletonVAE, self).__init__()
        self.train_batch_nb = self.hparams.batch_size
        self.vaL_batch_nb = 1
        self.latent_dim = self.hparams.latent_dim
        self.input_size = self.hparams.input_size
        self.cond_f_dims = self.hparams.cond_f_dims
        self.loss_weight = self.hparams.kld_loss_weight  # will multiply kld loss
        self.input_size = self.hparams.input_size
        self.cond_size = self.hparams.cond_size
        in_channels = self.hparams.in_channels

        # TODO it may be better to preprocess condition and input as same size.
        # input is C,H*W, make the size same by Linear
        self.res_in_channels = 64
        self.embed_condition = nn.Linear(
            self.cond_f_dims, self.input_size * self.input_size)
        # this is for feature fusion
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        if self.hparams.hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        if self.hparams.extract_layers is None:
            extract_layers = [2, 2, 2, 2]  # for condition feature extractor
        enc_in_channels = in_channels + 1  # add condition embeded
        ##########  encoder, img_size/2 for each conv, but feature_num *2 #####
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(enc_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            enc_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # TODO here is default 256. 256/64=4
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)

        ######### decoder #########
        modules = []
        self.decoder_input = nn.Linear(
            self.latent_dim + self.cond_f_dims, hidden_dims[-1] * 4)
        hidden_dims.reverse()  # list order reverse.
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        # final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

        ########### feature extractor for condition image #########
        af_size = math.ceil(self.cond_size / 32 - 7 + 1)
        self.feature_extracter = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False),
            # size/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # size/2
            self._make_layer(block, 64, extract_layers[0]),
            self._make_layer(
                block,
                128,
                extract_layers[1],
                stride=2),
            # size/2
            self._make_layer(
                block,
                256,
                extract_layers[2],
                stride=2),
            # size/2
            self._make_layer(
                block,
                512,
                extract_layers[2],
                stride=2),
            # size/2
            nn.AvgPool2d(7, stride=1),
        )
        # transfer feature map b,512,h,w -> b,cond_f_dims
        self.feature_mapping = nn.Linear(
            512 * block.expansion * af_size * af_size,
            self.cond_f_dims)

    def forward(self, x, c):  # input, condition
        # TODO now encoder-decoder structure, whether to use U-net
        # AND where should we add the condition? beginning or mid-term. NOW FOR INPUTS
        # here is flatten for c,w*h, then conv, relu

        cond_vector = self.feature_extract(c)  # maybe 64 or 128
        embedded_condition = self.embed_condition(cond_vector)
        embedded_condition = embedded_condition.view(
            -1, self.input_size, self.input_size).unsqueeze(1)
        embedded_input = self.embed_data(x)

        x_ = torch.cat([embedded_input, embedded_condition],
                       dim=1)  # concat on channel dim

        mu, log_var = self.encode(x_)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, cond_vector], dim=1)  # TODO data augment
        return [self.decode(z), mu, log_var]

    def encode(self, x):
        '''
        forward subpart. encodes the input through self.encoder. generate latent variable.
        :param x:  [N,C,H,W]
        :return: latent variables
        '''
        result = self.encoder(x)
        # [N x C x H x W] -> [N,CHW]
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def feature_extract(self, c):
        '''
        use pretrained model to extract information of condition image.
        :param c: condition image.
        :return: feature extracted
        '''
        # suppose c [b,C,H,W]
        feature_map = self.feature_extracter(c)
        feature_map = torch.flatten(feature_map, start_dim=1)
        feature_vector = self.feature_mapping(feature_map)
        return feature_vector

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _make_layer(self, block, channels, blocks, stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        # self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.res_in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.res_in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = list()
        layers.append(
            block(
                self.res_in_channels,
                channels,
                stride,
                downsample))
        self.res_in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.res_in_channels, channels))

        return nn.Sequential(*layers)

    def my_loss(self, infer_out, y):
        # TODO construct loss, y_hat is the predicted
        recons = infer_out[0]  # output
        mu = infer_out[1]
        log_var = infer_out[2]
        recons_loss = F.mse_loss(recons, y)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.loss_weight * kld_loss
        return loss, recons_loss, -kld_loss

    def training_step(self, batch, batch_nb):
        x, c, y = batch
        y_hat, mu, log_var = self.forward(x, c)
        infer_out = [y_hat, mu, log_var]
        loss, recons_loss, kld_loss = self.my_loss(infer_out, y)
        return {
            'Train_loss': loss,
            'Train_Reconstruction_Loss': recons_loss,
            'Train_KLD': -kld_loss}

    def validation_step(self, batch, batch_nb):
        x, c, y = batch
        y_hat, mu, log_var = self.forward(x, c)
        infer_out = [y_hat, mu, log_var]
        loss, recons_loss, kld_loss = self.my_loss(infer_out, y)
        return {
            'Val_loss': loss,
            'Val_Reconstruction_Loss': recons_loss,
            'Val_KLD': -kld_loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['Val_loss'] for x in outputs]).mean()
        avg_recon_loss = torch.stack(
            [x['Val_Reconstruction_Loss'] for x in outputs]).mean()
        avg_kld_loss = torch.stack([x['Val_KLD'] for x in outputs]).mean()
        return {
            'avg_val_loss': avg_loss,
            'avg_val_recon_Loss': avg_recon_loss,
            'avg_val_KLD': -avg_kld_loss}

    def configure_optimizers(self):
        # TODO if we want to finetune, add para.requires_grad = False in the
        # separent part of model
        return [
            torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad,
                    self.parameters()),
                lr=self.hparams.lr)]

    def __dataloader(self, train):
        # init data generators
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        dataset = SkeletonTrainDataset(
            self.hparams.dataset_folder,
            None,
            None,
            transform=transform) if train else SkeletonValDataset(
            self.hparams.dataset_folder)  # TODO hparams input

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @ptl.data_loader
    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    @ptl.data_loader
    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=False)

    @ptl.data_loader
    def test_dataloader(self):
        log.info('Test data loader called.')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])
        # hparams: in_channels,out_channels,latent_dim,hidden_dims,extract_layers,input_size,cond_size, kld_loss_weight,
        #                  cond_f_dims=128,block=BasicBlock
        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)
        # TODO change params
        # network params
        # namely, input num of key points
        parser.add_argument('--in_channels', default=18, type=int)
        # namely, output num of key points
        parser.add_argument('--out_channels', default=21, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument(
            '--hidden_dims',
            default=None,
            type=int)  # default [32,64,128,256,512]
        parser.add_argument(
            '--latent_dim',
            default=128,
            type=int)  # 128 or 256 not sure
        parser.add_argument(
            '--extract_layers',
            default=None,
            type=int)  # default [2,2,2,2]
        # input_size of heatmap or skeleton
        parser.add_argument('--input_size', default=256, type=int)
        parser.add_argument(
            '--cond_size',
            default=256,
            type=int)  # condition img size
        # not sure how to set weights between KLd and recon loss
        parser.add_argument('--kld_loss_weight', default=0.2, type=float)
        # dims of feature extractor for cond images
        parser.add_argument('--cond_f_dims', default=128, type=int)
        parser.add_argument(
            '--dataset_folder',
            default="datasets",
            type=str,
            help="upper folder for train and val data")
        parser.add_argument(
            '--transformer_flag',
            default=True,
            type=bool,
            help="whether to use transformer")

        # training params (opt)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=16, type=int)
        return parser

# ==================Definition Start - Stage 12======================


class Generator(nn.Module):
    def __init__(self):
        # TODO change model
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * args.dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * args.dim, 2 * args.dim, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * args.dim, args.dim, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(args.dim, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # TODO change model
        output = self.preprocess(input)
        output = output.view(-1, 4 * args.dim, 4, 4)
        # print output.size()
        output = self.block1(output)
        # print output.size()
        output = output[:, :, :7, :7]
        # print output.size()
        output = self.block2(output)
        # print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        # print output.size()
        return output.view(-1, args.output_dim)


class Discriminator(nn.Module):
    def __init__(self):
        # TODO change model
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, args.dim, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*args.dim),
            nn.ReLU(True),
            nn.Conv2d(args.dim, 2 * args.dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2 * args.dim, 4 * args.dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * args.dim, 1)

    def forward(self, input):
        # TODO change model
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * args.dim)
        out = self.output(out)
        return out.view(-1)
