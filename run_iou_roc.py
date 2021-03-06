import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision
import seaborn as sns

from utils.data import prepare_dataloader
from utils.train import plot_roc_iou
from utils.misc import log_print
from models.enhanced import CloudSegment

sns.set()

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
    batch_size = 4
    img_size = (int(4*64), int(6*64))
    classes = ['sugar','flower','fish','gravel']
    iou_threshold = 0.5
    grayscale = False
    drop_empty = True

    mask_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor()
                                                    ])

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_augmentations = None

    log_print('Preparing dataloaders', logging)
    _, validation_dataloader, valid_dl_no_empty, _ = prepare_dataloader(train_image_filepath = train_image_filepath,
                                                                        test_image_filepath = test_image_filepath,
                                                                        df_filepath = df_filepath,
                                                                        seed = seed,
                                                                        train_transform = train_transform,
                                                                        test_transform = test_transform,
                                                                        mask_transform= mask_transform,
                                                                        size = img_size,
                                                                        batch_size = batch_size, 
                                                                        label = classes, 
                                                                        data_augmentations = data_augmentations, 
                                                                        grayscale = grayscale,
                                                                        drop_empty = drop_empty
                                                                        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','final_flowerUnet_EfficientNetEncoder_current_model.pth'), map_location = device)

    segmentation_model = CloudSegment(classifier_path = os.path.join(os.getcwd(),'weights','classifier.pth'),
                                        sugar_path = os.path.join(os.getcwd(),'weights','final_sugarUnet_EfficientNetEncoder_current_model.pth'),
                                        flower_path = os.path.join(os.getcwd(),'weights','final_flowerUnet_EfficientNetEncoder_current_model.pth'),
                                        fish_path = os.path.join(os.getcwd(),'weights','final_fishUnet_EfficientNetEncoder_current_model.pth'),
                                        gravel_path = os.path.join(os.getcwd(),'weights','final_gravelUnet_EfficientNetEncoder_current_model.pth'),
                                        device=device)
    
    plot_roc_iou(dataloader_list = [validation_dataloader],
                dataloader_name_list = ['Full Validation Set'],
                model = segmentation_model,
                classes = classes,
                batch_samples = 2,
                device = torch.device("cpu"),
                logger = logging,
                plots_save_path = os.path.join(os.getcwd(),'roc_iou_plots'))