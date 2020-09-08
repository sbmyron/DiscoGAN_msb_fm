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
        self.conv0 = nn.Conv2d(3, 3, (5,5), (1,5), (2,0), bias=False)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True) 
 
        self.conv5 = nn.Conv2d(64 * 8, 64 * 16, 3, 3, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(64 * 16)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
   
        self.conv6 = nn.Conv2d(64 * 16, 1, 3, 3, 0, bias=False)

    def forward(self, input):
    	#print('========Input shape:', input.shape)
        conv0 = self.conv0( input )
        relu0 = self.relu0( conv0 )  
        #print('========conv0 shape:', relu0.shape)		

        conv1 = self.conv1( relu0 )
        bn1 = self.bn1( conv1 )
        relu1 = self.relu1( bn1 ) 
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
 
     #    conv6 = self.conv6( relu5 )
     #    bn6 = self.bn6( conv6 )
     #    relu6 = self.relu6( bn6 )
    	# print('========conv6 shape:', relu6.shape)
 
     #    conv7 = self.conv7( relu6 )
     #    bn7 = self.bn7( conv7 )
     #    relu7 = self.relu7( bn7 )
    	# print('========conv7 shape:', relu7.shape)

        conv6 = self.conv6( relu5 )
    	#print('========conv8 shape:', conv8.shape)

        return torch.sigmoid( conv6 ), [relu3, relu4, relu5]

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
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 64 * 2, 4, 2, 0, bias=False), 
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 0, bias=False), 
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False), 
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64 * 8, 64 * 16, 3, 3, 0, bias=False), 
                nn.BatchNorm2d(64 * 16),
                nn.LeakyReLU(0.2, inplace=True),
   
                # # batch x 64*8 x 3 x 3	
                # #Perhaps add a 1x1x1000 vector??

                nn.ConvTranspose2d(64 * 16, 64 * 8, 3, 3, 0, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True), 
                #batch x 64*4 x 9 x 9
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 0, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True), 
                #batch x 64*2 x 20 x 20
                nn.ConvTranspose2d(64 * 4,  64 * 2 , 4, 2, 2, output_padding = 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True), 
                #batch x 64 x 39 x 39 
                nn.ConvTranspose2d(64 * 2,  64  , 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True), 
                #batch x 64 x 78 * 78
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                #batch x 64 x 156 * 156
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.main( input )


