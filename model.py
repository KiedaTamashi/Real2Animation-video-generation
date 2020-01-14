import os, sys
sys.path.append(os.getcwd())
from settings import args
import torch.nn as nn




# ==================Definition Start======================
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




