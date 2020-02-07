import os, sys
sys.path.append(os.getcwd())
from settings import args
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch

import pytorch_lightning as ptl
import logging as log
from argparse import ArgumentParser
from collections import OrderedDict
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

from dataset import SkeletonTrainDataset,SkeletonValDataset
# ==================Definition Start - Stage 1======================
class skeletonVAE(ptl.LightningModule):
    def __init__(self):
        super(skeletonVAE, self).__init__()
        # not the best model...
        #TODO construct model structure
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x, c):#input, condition
        # TODO construct forward
        # here is flatten for c,w*h, then conv, relu
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def my_loss(self, y_hat, y):
        # TODO construct loss, y_hat is the predicted
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        x, c, y = batch
        y_hat = self.forward(x,c)
        return {'loss': self.my_loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, c, y = batch
        y_hat = self.forward(x, c)
        return {'val_loss': self.my_loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        #TODO if we want to finetune, add para.requires_grad = False in the separent part of model
        return [torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.02)]

    def __dataloader(self, train):
        # init data generators
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))])
        #TODO construct datasets
        dataset = SkeletonTrainDataset if train else SkeletonValDataset

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
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'datasets'), type=str)

        # training params (opt)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=32, type=int)
        return parser

# ==================Definition Start - Stage 12======================
class Generator(nn.Module):
    def __init__(self):
        #TODO change model
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*args.dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*args.dim, 2*args.dim, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*args.dim, args.dim, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(args.dim, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #TODO change model
        output = self.preprocess(input)
        output = output.view(-1, 4*args.dim, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, args.output_dim)

class Discriminator(nn.Module):
    def __init__(self):
        # TODO change model
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, args.dim, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*args.dim),
            nn.ReLU(True),
            nn.Conv2d(args.dim, 2*args.dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*args.dim, 4*args.dim, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*args.dim, 1)

    def forward(self, input):
        # TODO change model
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*args.dim)
        out = self.output(out)
        return out.view(-1)

