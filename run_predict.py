import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader
from utils.train import validate_and_plot

if __name__ == '__main__':
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    logging.basicConfig(filename= os.path.join(log_file_path,"prediction_log_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    # Return relevant dataloaders from whatever matthew's function is
    cwd = os.getcwd()
    train_image_filepath = os.path.join(cwd,'data','train_images')
    test_image_filepath = os.path.join(cwd,'data','test_images')
    df_filepath = os.path.join(cwd,'data','train.csv')
    seed = 2
    batch_size = 1
    img_size = (6*64, 9*64)

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataloader, validation_dataloader, _ = prepare_dataloader(train_image_filepath,
                                                                    test_image_filepath,
                                                                    df_filepath,
                                                                    seed,
                                                                    train_transform,
                                                                    test_transform,
                                                                    size = img_size,
                                                                    batch_size = batch_size,
                                                                    shuffle_val_dataloader = True)

    segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','densenet169_best_model.pth'))

    validate_and_plot(validation_dataloader = validation_dataloader,
                        model = segmentation_model,
                        num_plots = 20,
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        logger = logging,
                        plots_save_path = os.path.join(os.getcwd(),'prediction_plots'))

    validate_and_plot(validation_dataloader = train_dataloader,
                        model = segmentation_model,
                        num_plots = 20,
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        logger = logging,
                        plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
                        prefix='Train')
