import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import FeatureAblation


if __name__ == "__main__":
    model = torch.load('weights/classifier.pth')
    model.eval()
    torch.manual_seed(123)
    np.random.seed(123)

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    img = Image.open('data/train_images/0a1b596.jpg')
    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0).to('cuda:0')
    output = model(input)

    idx_to_labels = ['flower', 'gravel', 'sugar', 'fish']

    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()

    predicted_label = idx_to_labels[pred_label_idx.item()]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    integrated_gradients = IntegratedGradients(model)
    # attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)

    # default_cmap = LinearSegmentedColormap.from_list('custom blue', 
    #                                                 [(0, '#ffffff'),
    #                                                 (0.25, '#000000'),
    #                                                 (1, '#000000')], N=256)

    # pic = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
    #                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
    #                             method='heat_map',
    #                             cmap=default_cmap,
    #                             show_colorbar=True,
    #                             sign='positive',
    #                             outlier_perc=1)

    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(input, n_samples=5, nt_type='smoothgrad_sq', target=pred_label_idx)
    pic = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        cmap=default_cmap,
                                        show_colorbar=True)

    pic[0].savefig('model_viz.png')