import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader, get_augmentations
from utils.train_modified import train_model
from utils.misc import upload_google_sheets, get_module_name, log_print
from models.auxillary import BinaryFocalLoss
from models.unet import Unet
from models.enhanced import CloudSegment

if __name__ == '__main__':
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    start_time = time.ctime()
    logging.basicConfig(filename= os.path.join(log_file_path,"training_log_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
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
    start_lr = 1e-4
    classes = ['sugar','flower','fish','gravel']
    iou_threshold = 0.5
    total_epochs = 10
    grayscale = False
    drop_empty = True
    loss_args = dict(
        beta = 1,
        gamma = 2.
    )
    aux_params = dict(
        classes = len(classes),
        activation = 'sigmoid'
    )

    mask_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor()
                                                    ])

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_augmentations = get_augmentations(img_size)

    train_dataloader, validation_dataloader, valid_dl_no_empty, test_dataloader = prepare_dataloader(train_image_filepath = train_image_filepath,
                                                                                test_image_filepath = test_image_filepath,
                                                                                df_filepath =df_filepath,
                                                                                seed = seed,
                                                                                train_transform = train_transform,
                                                                                test_transform = test_transform,
                                                                                mask_transform= mask_transform,
                                                                                size = img_size,
                                                                                batch_size = batch_size, 
                                                                                label = classes, 
                                                                                data_augmentations = data_augmentations, 
                                                                                grayscale = grayscale,
                                                                                drop_empty = drop_empty,
                                                                                return_labels = True
                                                                                )

    # Define Model
    # segmentation_model = CloudSegment(classifier_path = os.path.join(os.getcwd(),'weights','classifier.pth'),
    #                                     sugar_path = os.path.join(os.getcwd(),'weights','final_sugarUnet_EfficientNetEncoder_current_model.pth'),
    #                                     flower_path = os.path.join(os.getcwd(),'weights','final_flowerUnet_EfficientNetEncoder_current_model.pth'),
    #                                     fish_path = os.path.join(os.getcwd(),'weights','final_fishUnet_EfficientNetEncoder_current_model.pth'),
    #                                     gravel_path = os.path.join(os.getcwd(),'weights','final_gravelUnet_EfficientNetEncoder_current_model.pth'))

    model_save_prefix = 'cloud_segmentator'
    # segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','baseline.pth'))
    segmentation_model = Unet('efficientnet-b3', encoder_weights='imagenet',classes=len(classes), activation='sigmoid', decoder_attention_type='scse', aux_params = aux_params)

    params = dict(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose = True,
    )

    # Define Loss and Accuracy Metric
    loss = [smp.utils.losses.DiceLoss(beta = loss_args['beta']) + BinaryFocalLoss(gamma = loss_args['gamma']),BinaryFocalLoss(gamma = loss_args['gamma'])]
    metrics = [
        smp.utils.metrics.IoU(threshold=iou_threshold),
        smp.utils.metrics.Precision(threshold=iou_threshold),
        smp.utils.metrics.Recall(threshold=iou_threshold)
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params= segmentation_model.parameters(), lr=start_lr),
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1)

    losses, metric_values, best_epoch, confusion_matrices = train_model(train_dataloader = train_dataloader,
                                                            validation_dataloader_list = [validation_dataloader],
                                                            model = segmentation_model,
                                                            loss = loss,
                                                            metrics = metrics,
                                                            optimizer = optimizer,
                                                            scheduler = scheduler,
                                                            batch_size = batch_size,
                                                            num_epochs = total_epochs,
                                                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                                            classes = classes,
                                                            logger = logging,
                                                            verbose = True,
                                                            # only_validation = True,
                                                            model_save_path = os.path.join(os.getcwd(),'weights'),
                                                            model_save_prefix = model_save_prefix,
                                                            plots_save_path = os.path.join(os.getcwd(),'plots')
                                                            )
    log_print(f'Validation', logging)