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

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
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
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.softmax  = nn.Softmax(dim=-1)
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
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
       
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        return out

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
    To fit into memory requirements, we have to downsample and then upsample the 
    '''
    def __init__(self, in_channels, down_sample_channel_factor = 8, downsample_wh_factor = 4):
        super(PAM_CAM_Module, self).__init__()
        self.pre_mod = nn.Conv2d(in_channels, max(in_channels//down_sample_channel_factor,1), kernel_size=downsample_wh_factor, padding=0, stride=downsample_wh_factor)
        self.pam = PAM_CAM_Layer(max(in_channels//down_sample_channel_factor,1))
        self.cam = PAM_CAM_Layer(max(in_channels//down_sample_channel_factor,1), use_pam = False)
        self.post_mod = nn.ConvTranspose2d(max(in_channels//down_sample_channel_factor,1), in_channels, kernel_size=downsample_wh_factor, padding=0, stride=downsample_wh_factor)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        preX = x
        x = self.pre_mod(x)
        x = self.pam(x) + self.cam(x)
        x = self.post_mod(x)
        return preX + self.gamma*x