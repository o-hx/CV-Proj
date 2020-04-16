import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The position and channel attention modules are from https://arxiv.org/pdf/1906.02849.pdf
Code is adapted, and modified from https://github.com/sinAshish/Multi-Scale-Attention
'''

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_atrous_conv_down = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation = 2)
        self.query_atrous_conv_up = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, dilation = 2)
        self.key_conv_down = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, dilation = 4)
        self.key_conv_up = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, dilation = 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        pooled_x = torch.mean(x, dim = 1, keepdim=True) # Take the average pooling across the channels
        query_result = self.query_atrous_conv_up(self.query_atrous_conv_down(pooled_x))
        key_result = self.key_conv_up(self.key_conv_down(pooled_x))
        weights = self.sigmoid(query_result + key_result)
        
        return x + x*weights

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()

        self.chanel_in = in_dim
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.down_conv1 = nn.Conv1d(1, 1, kernel_size=3, dilation=2)
        self.down_conv2 = nn.Conv1d(1, 1, kernel_size=3, dilation=2)
        self.up_conv2 = nn.ConvTranspose1d(1, 1, kernel_size=3, dilation=2)
        self.up_conv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, dilation=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        pooled_x = self.spatial_pool(x).view(m_batchsize,1,-1)
        pooled_x = self.down_conv1(pooled_x)
        pooled_x = self.down_conv2(pooled_x)
        pooled_x = self.up_conv2(pooled_x)
        pooled_x = self.up_conv1(pooled_x)
        weights = self.sigmoid(pooled_x).view(m_batchsize, C, 1,1)

        return x + x*weights

class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention
    
    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, use_pam = True):
        super(PAM_CAM_Layer, self).__init__()
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.attn(x)

class PAM_CAM_Module(nn.Module):
    '''
    To fit into memory requirements, we have to downsample and then upsample the result
    '''
    def __init__(self, in_channels, down_sample_channel_factor = 8, downsample_wh_factor = 4):
        super(PAM_CAM_Module, self).__init__()
        self.pam = PAM_CAM_Layer(in_channels)
        self.cam = PAM_CAM_Layer(in_channels, use_pam = False)

    def forward(self, x):
        pamx = self.pam(x)
        camx = self.cam(x)
        x = pamx + camx
        return x