import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The position and channel attention modules are from https://arxiv.org/pdf/1906.02849.pdf
Code is adapted, and extensively modified from https://github.com/sinAshish/Multi-Scale-Attention. All BMM features have been removed to reduce memory usage
'''

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.encode_decode_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride = 2)
        )
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
        key_result = self.encode_decode_2(pooled_x)
        weights = self.sigmoid(key_result)
        
        return x + x*weights

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()

        self.chanel_in = in_dim
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.encode_decode = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
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
        weights = self.encode_decode(pooled_x)
        weights = weights.view(m_batchsize, C, 1,1)

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