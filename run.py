import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader
from utils.train import train_model

if __name__ == '__main__':
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    logging.basicConfig(filename= os.path.join(log_file_path,"training_log_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    # Return relevant dataloaders from whatever matthew's function is
    cwd = os.getcwd()
    train_image_filepath = os.path.join(cwd,'data','train_images')
    test_image_filepath = os.path.join(cwd,'data','test_images')
    df_filepath = os.path.join(cwd,'data','train.csv')
    seed = 2
    batch_size = 8
    img_size = (int(4*64), int(6*64))
    model_save_prefix = 'dl_efficientb2_'

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_augmentations = [torchvision.transforms.RandomHorizontalFlip(p= 1), 
                          torchvision.transforms.RandomVerticalFlip(p= 1)]

    train_dataloader, validation_dataloader, test_dataloader = prepare_dataloader(train_image_filepath,
                                                                                test_image_filepath,
                                                                                df_filepath,
                                                                                seed,
                                                                                train_transform,
                                                                                test_transform,
                                                                                size = img_size,
                                                                                batch_size = batch_size, 
                                                                                label = None, 
                                                                                data_augmentations = data_augmentations)

    # Define Model
    # segmentation_model = smp.Unet('densenet169', encoder_weights='imagenet',classes=4, activation='sigmoid', decoder_attention_type = 'scse')
    segmentation_model = smp.PAN('densenet169', encoder_weights='imagenet',classes=4, activation='sigmoid')
    # segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','densenet169_best_model - Copy.pth'))

    # Freeze the encoder parameters for now (just train the decoder)
    # for param in segmentation_model.encoder.parameters():
    #     param.requires_grad = False

    params = dict(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose = True,
    )

    # Define Loss and Accuracy Metric
    loss = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params= segmentation_model.parameters(), lr=0.001),
    ])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0)

    train_model(train_dataloader = train_dataloader,
                validation_dataloader = validation_dataloader,
                model = segmentation_model,
                loss = loss,
                metrics = metrics,
                optimizer = optimizer,
                scheduler = scheduler,
                batch_size = batch_size,
                num_epochs = 10,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = logging,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'weights'),
                model_save_prefix = model_save_prefix,
                plots_save_path = os.path.join(os.getcwd(),'plots')
                )