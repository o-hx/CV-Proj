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
    batch_size = 16
    img_size = (287, 287)
    start_lr = 0.001
    classes = ['gravel']
    total_epochs = 1
    k = 2

    train_dataloader, validation_dataloader = get_dataloader(df_filepath = df_filepath,
                                                train_image_filepath = train_image_filepath,
                                                img_size = img_size,
                                                label = classes[0],
                                                normalise = True,
                                                batch_size = batch_size
                                                )

    org_train_dataloader, org_validation_dataloader = get_dataloader(df_filepath = df_filepath,
                                                                        train_image_filepath = train_image_filepath,
                                                                        img_size = img_size,
                                                                        label = classes[0],
                                                                        normalise = False,
                                                                        batch_size = batch_size
                                                                        )


    autoencoder = Autoencoder()
    model_save_prefix = classes[0] + '_' + get_module_name(autoencoder)

    params = dict(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose = True,
    )


    loss = smp.utils.losses.MSELoss()
    metrics = [
        MSE()
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params= autoencoder.parameters(), lr=start_lr),
    ])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0)

    losses, metric_values, best_epoch, model = train_model(train_dataloader = train_dataloader,
                                                            validation_dataloader_list = [validation_dataloader],
                                                            model = autoencoder,
                                                            loss = loss,
                                                            metrics = metrics,
                                                            optimizer = optimizer,
                                                            scheduler = scheduler,
                                                            batch_size = batch_size,
                                                            num_epochs = total_epochs,
                                                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                                            classes = classes,
                                                            logger = logging,
                                                            autoencoder = True,
                                                            verbose = True,
                                                            model_save_path = os.path.join(os.getcwd(),'weights'),
                                                            model_save_prefix = model_save_prefix,
                                                            plots_save_path = os.path.join(os.getcwd(),'plots')
                                                            )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # autoencoder = torch.load(os.path.join(os.getcwd(),'weights','fish_Autoencodercurrent_model.pth'), map_location = device)
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD([
        dict(params= autoencoder.parameters(), lr=0.01, momentum = 0.9),
    ])

    autoencoder = cluster(autoencoder,
                        train_dataloader,
                        k = k,
                        criterion = loss,
                        optimizer = optimizer,
                        scheduler = None,
                        total_epochs = 10,
                        logger = logging,
                        model_save_path = os.path.join(os.getcwd(),'weights'),
                        model_save_prefix = classes[0],
                        )

    plot_clusters(model = autoencoder,
                k = k,
                clas = classes[0],
                dataloader = train_dataloader,
                batch_size = batch_size,
                original_dataloader = org_train_dataloader,
                plots_save_path = os.path.join(os.getcwd(),'clustering_plots'),
                logger = logging
                )

    log_print(f'Completed Clustering', logging)