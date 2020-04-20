import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader, get_augmentations
from utils.train import train_model, cluster, plot_clusters
from utils.misc import upload_google_sheets, get_module_name, log_print
from utils.tsne import get_dataloader
from models.autoencoder import Autoencoder
from models.auxillary import MSE

if __name__ == '__main__':
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    start_time = time.ctime()
    logging.basicConfig(filename= os.path.join(log_file_path,"autoencode_log_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    # Return relevant dataloaders from whatever matthew's function is
    cwd = os.getcwd()
    train_image_filepath = os.path.join(cwd,'data','train_images')
    df_filepath = os.path.join(cwd,'data','train.csv')
    seed = 2
    batch_size = 32
    img_size = (287, 287)
    start_lr = 0.001
    classes = ['flower']
    total_epochs = 10
    k = 5

    train_dataloader, validation_dataloader = get_dataloader(df_filepath = df_filepath,
                                                train_image_filepath = train_image_filepath,
                                                img_size = img_size,
                                                label = classes[0],
                                                normalise = True,
                                                batch_size = batch_size
                                                )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder = torch.load(os.path.join(os.getcwd(),'weights',f'{classes[0]}_Autoencodercurrent_model.pth'), map_location = device)

    autoencoder.eval()
    with torch.no_grad():
        for _, data in enumerate(train_dataloader):
            x, _ = data
            x = x.to(device)
            _ = autoencoder.forward(x)

            latent = autoencoder.latent.shape.detach().cpu().numpy()
