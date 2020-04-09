import torch

from segmentation_models_pytorch.utils.base import Loss
from segmentation_models_pytorch.utils.functional import _take_channels, _threshold

# Function for loss
def binary_focal_loss_with_logits(logits, y_pred, y, alpha, gamma = 0., threshold = None, ignore_channels = None, reduction = 'mean'):
    '''
    Implementation of focal loss for binary classification
    Adapted from binary cross entropy loss
    See https://arxiv.org/pdf/1708.02002.pdf for further details
    '''
    # y_pred (tensor, shape = B, classes, H, W): Predicted Tensor
    # y (tensor, shape = B, classes, H, W): Ground Truth Tensor. Must have same shape as predicted tensor

    # alpha: A tensor of shape (# classes)
    # gamma: A float which is proportional to the weight that harder samples should exert on the loss. If gamma = 1, the function is a BCE loss

    alpha = alpha.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    y_pred = _threshold(y_pred, threshold=threshold)
    y_pred, y = _take_channels(y_pred, y, ignore_channels=ignore_channels)
    #element_wise_weighted_BCE = -alpha*(y*torch.pow(1-y_pred, gamma)*torch.log(y_pred) + (1-y)*torch.pow(y_pred, gamma)*torch.log(1-y_pred)) # This is the non-logits version, which is less stable

    weight_a = alpha * (1 - y_pred) ** gamma * y
    weight_b = alpha * y_pred ** gamma * (1 - y)
    element_wise_weighted_BCE = (torch.log1p(torch.exp(-torch.abs(logits))) + torch.nn.functional.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    if reduction == 'mean':
        return torch.mean(element_wise_weighted_BCE) # Take the mean across all elements i.e. pixels
    elif reduction == 'sum':
        return element_wise_weighted_BCE.sum()
    else:
        raise NotImplementedError(f'{reduction} not yet implemented')

# Loss
class BinaryFocalLoss(Loss):
    def __init__(self, gamma = 2., alpha = None, reduction = 'mean', eps = 1e-7, ignore_channels = None, multiplier = 1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha
        self.eps = float(eps)
        self.ignore_channels = None
        self.reduction = reduction
        self.multiplier = multiplier
    
    def forward(self, y_pred, y):
        alpha = torch.ones(y_pred.shape[1]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) if self.alpha is None else self.alpha
        y_pred = y_pred + self.eps
        logits = torch.log(y_pred/(1-y_pred))

        assert alpha.shape[0] == y_pred.shape[1] # Number of classes must be same as number of class weights
        assert len(alpha.shape) == 1 # Alpha must be a tensor of length 1

        return self.multiplier*binary_focal_loss_with_logits(
            logits, y_pred, y, alpha,
            gamma = self.gamma,
            threshold=None,
            ignore_channels=self.ignore_channels,
            reduction=self.reduction
        )
