import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import ipdb

import numpy as np

kernel_sizes = [4,3,3]
strides = [2,2,1]
paddings=[0,0,1]

latent_dim = 300

class Discriminator(nn.Module):
    def __init__(
            self,
            ):

        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 2, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(32 * 2, 32 * 4, 4, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(32 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(32 * 4, 32 * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(32 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
 
        self.conv5 = nn.Conv2d(32 * 8, 32 * 8, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(32 * 8)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
  

        self.conv8 = nn.Conv2d(32 * 8, 1, 11, 1, 0, bias=False)

    def forward(self, input):
    	#print('========Input shape:', input.shape)
        conv1 = self.conv1( input )
        relu1 = self.relu1( conv1 ) 

    	#print('========conv1 shape:', relu1.shape)
        conv2 = self.conv2( relu1 )
        bn2 = self.bn2( conv2 )
        relu2 = self.relu2( bn2 )
    	#print('========conv2 shape:', relu2.shape)

        conv3 = self.conv3( relu2 )
        bn3 = self.bn3( conv3 )
        relu3 = self.relu3( bn3 )
    	#print('========conv3 shape:', relu3.shape)

        conv4 = self.conv4( relu3 )
        bn4 = self.bn4( conv4 )
        relu4 = self.relu4( bn4 )
    	#print('========conv4 shape:', relu4.shape)

        conv5 = self.conv5( relu4 )
        bn5 = self.bn5( conv5 )
        relu5 = self.relu5( bn5 )
    	#print('========conv5 shape:', relu5.shape)
  
        conv8 = self.conv8( relu5 )
    	#print('========conv8 shape:', conv8.shape)

        return torch.sigmoid( conv8 ), [relu3, relu4, relu5]

class Generator(nn.Module):
    def __init__(
            self,
            extra_layers=False
            ):

        super(Generator, self).__init__()

        if extra_layers == True:
            self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 8, 100, 4, 1, 0, bias=False),
                nn.BatchNorm2d(100),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(    64,      3, 4, 2, 1, bias=False),
                nn.Sigmoid()
            )


        if extra_layers == False:
            self.main = nn.Sequential(
                nn.Conv2d(3, 24, 4, 2, 2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(24, 24 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(24 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(24 * 2, 24 * 4, 4, 2, 2, bias=False),
                nn.BatchNorm2d(24 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(24 * 4, 24 * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(24 * 8),
                nn.LeakyReLU(0.2, inplace=True),

                #My extra layers
                nn.Conv2d(24 * 8, 24 * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(24 * 16),
                nn.LeakyReLU(0.2, inplace=True), 

                nn.Conv2d(24 * 16, 24 * 32, 4, 2, 2, bias=False),
                nn.BatchNorm2d(24 * 32),
                nn.LeakyReLU(0.2, inplace=True), 

                nn.Conv2d(24 * 32, 24 * 64, 4, 2, 2, bias=False),
                nn.BatchNorm2d(24 * 64),
                nn.LeakyReLU(0.2, inplace=True),  

                # # batch x 1024 x 4 x 4	
                # #Perhaps add a 1x1x1000 vector??

                nn.ConvTranspose2d(24 * 64, 24 * 32, 3, 1, 0, bias=False),
                nn.BatchNorm2d(24 * 32),
                nn.ReLU(True), 
                #batch x 512 x 6 x 6  
                nn.ConvTranspose2d(24 * 32, 24 * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(24 * 16),
                nn.ReLU(True), 
                #batch x 256 x 12 x 12
                nn.ConvTranspose2d(24 * 16,  24 * 8, 4, 2, 2, bias=False),
                nn.BatchNorm2d(24 * 8),
                nn.ReLU(True),
                #batch x 128 x 22 x 22
                nn.ConvTranspose2d(24 * 8,   24 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(24 * 4),
                nn.ReLU(True),
                #batch x 64 x 44 x 44
                nn.ConvTranspose2d(24 * 4,   24 * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(24 * 2),
                nn.ReLU(True),
                #batch x 32 x 87 x 87
                nn.ConvTranspose2d(24 * 2,   24    , 4, 2, 1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(True),
                #batch x 16 x 174 x 174
                nn.ConvTranspose2d(24    ,   3     , 3, 2, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.main( input )


