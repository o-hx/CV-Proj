import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader
from utils.train import validate_and_plot, log_print

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
    img_size = (int(4*64), int(6*64))
    classes = ['fish','gravel']
    iou_threshold = 0.5
    total_epochs = 10
    grayscale = True

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_augmentations = [torchvision.transforms.RandomHorizontalFlip(p= 1),
                          torchvision.transforms.RandomVerticalFlip(p= 1)]

    log_print('Preparing dataloaders', logging)
    train_dataloader, validation_dataloader, _ = prepare_dataloader(train_image_filepath,
                                                                                test_image_filepath,
                                                                                df_filepath,
                                                                                seed,
                                                                                train_transform,
                                                                                test_transform,
                                                                                size = img_size,
                                                                                batch_size = batch_size, 
                                                                                label = classes,
                                                                                shuffle_train_dataloader = False,
                                                                                data_augmentations = data_augmentations, 
                                                                                grayscale = grayscale)

    log_print('Preparing original dataloaders', logging)
    train_transform_modified = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()])
    test_transform_modified = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size), torchvision.transforms.ToTensor()])
    train_dataloader_org, validation_dataloader_org, _ = prepare_dataloader(train_image_filepath,
                                                                                test_image_filepath,
                                                                                df_filepath,
                                                                                seed,
                                                                                train_transform_modified,
                                                                                test_transform_modified,
                                                                                size = img_size,
                                                                                batch_size = batch_size, 
                                                                                label = classes,
                                                                                shuffle_train_dataloader = False,
                                                                                equalise = False,
                                                                                grayscale = False)

    segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','DeepLabV3SENetEncoder_best_model.pth'))

    metrics = [
        smp.utils.metrics.IoU(threshold=iou_threshold),
    ]
    
    validate_and_plot(validation_dataloader = validation_dataloader,
                        validation_dataloader_org = validation_dataloader_org,
                        model = segmentation_model,
                        metrics = metrics,
                        classes = classes,
                        top_n = 20,
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        logger = logging,
                        threshold = iou_threshold,
                        plots_save_path = os.path.join(os.getcwd(),'prediction_plots'))

    # validate_and_plot(validation_dataloader = train_dataloader,
    #                     model = segmentation_model,
    #                     num_plots = 20,
    #                     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    #                     logger = logging,
    #                     plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
    #                     prefix='Train')
