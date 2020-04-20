import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from efficientnet_pytorch import EfficientNet

class Autoencoder(nn.Module):
    '''
    Define a generic autoencoder that takes in an image, compresses the image into a 1*1280 vector, and then outputs the image from that 1*20 vector
    For the encoder, we will use efficientnet b0
    There are no attention modules in the Autoencoder
    '''
    def __init__(self,
                efficient_net_type = 'efficientnet-b0',
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ):
        super(Autoencoder, self).__init__()

        self.device = device
        self.encoder = self.get_efficient_net(efficient_net_type)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=20,out_channels=18,kernel_size=3,stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(18),
            nn.ConvTranspose2d(in_channels=18,out_channels=15,kernel_size=3, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(15),
            nn.ConvTranspose2d(in_channels=15,out_channels=10,kernel_size=3, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(in_channels=10,out_channels=5,kernel_size=3, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(5),
            nn.ConvTranspose2d(in_channels=5,out_channels=3,kernel_size=3, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
        )
        self.clustering = False

    def get_efficient_net(self, efficient_net_type):
        en = EfficientNet.from_pretrained(efficient_net_type) # Outputs a (batch size, 1000) vector. Change to 1024 so we can reshape as 32*32
        en._fc = nn.Linear(1280, out_features = 1280, bias = True).to(self.device)
        return en

    def forward(self, x):
        x = self.encoder(x)
        if not self.clustering:
            self.latent = x.detach().cpu()
            x = x.view(-1,20,8,8)
            x = self.decoder(x)
            return x
        else:
            q = self.K_means(x)
            return q, x
    
    def init_kmeans_weights(self):
        self.k_means_weights = nn.Parameter(torch.Tensor(self.k,1280).uniform_(-1,1)*np.sqrt(2/(self.k+1280)), requires_grad=True).to(self.device)

    def K_means(self, x):
        '''
        K means layer that should be activated only after training

        returns qij, probability of xij being in a certain cluster
        '''
        batch_size, features = x.shape
        q = 1./(1.+torch.sum((x.unsqueeze(1) - self.k_means_weights)**2, axis = 2)/self.alpha)
        q = torch.pow(q, (self.alpha + 1.)/2.)
        q = q.view(batch_size, self.k)
        q = q/q.sum(axis = 1, keepdim=True)
        return q

    def switch_to_kmeans(self, k, alpha = 1.0):
        # Set encoder to eval only. Do not train the encoder anymore
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Initialise kmeans layer
        self.k = k
        self.alpha = 1.0
        self.clustering = True
        self.init_kmeans_weights()