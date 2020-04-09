import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision
import re

from utils.data import prepare_dataloader
from utils.train import train_model, log_print
from utils.sheets_upload import upload_google_sheets
from models import BinaryFocalLoss

def get_module_name(obj):
    try:
        return re.findall(r"[A-Za-z0-9]+'",str(type(obj)))[0].replace("'",'')
    except:
        return "Name not found"

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
    batch_size = 12
    img_size = (int(4*64), int(6*64))
    start_lr = 0.0005
    classes = ['fish']
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

    train_dataloader, validation_dataloader, test_dataloader = prepare_dataloader(train_image_filepath,
                                                                                test_image_filepath,
                                                                                df_filepath,
                                                                                seed,
                                                                                train_transform,
                                                                                test_transform,
                                                                                size = img_size,
                                                                                batch_size = batch_size, 
                                                                                label = classes, 
                                                                                data_augmentations = data_augmentations, 
                                                                                grayscale = grayscale)

    # Define Model
    segmentation_model = smp.Unet('se_resnet50', encoder_weights='imagenet',classes=len(classes), activation='sigmoid', decoder_attention_type='scse')
    # segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','dlv3_se_resnet50_current_model.pth'))
    model_save_prefix = get_module_name(segmentation_model) + '_' + get_module_name(segmentation_model.encoder) + '_'

    params = dict(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose = True,
    )

    # Define Loss and Accuracy Metric
    loss = BinaryFocalLoss(gamma=2., multiplier=1.) + smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=iou_threshold),
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params= segmentation_model.parameters(), lr=start_lr),
    ])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size =1, gamma=0.8)

    losses, metric_values, best_epoch = train_model(train_dataloader = train_dataloader,
                                        validation_dataloader = validation_dataloader,
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
                                        model_save_path = os.path.join(os.getcwd(),'weights'),
                                        model_save_prefix = model_save_prefix,
                                        plots_save_path = os.path.join(os.getcwd(),'plots')
                                        )
    log_print(f'Completed training and validation', logging)
    
    # Prepare dictionary to update to Google Sheets
    result = dict(
        model_name = get_module_name(segmentation_model),
        encoder = get_module_name(segmentation_model.encoder),
        image_size = str(img_size),
        batch_size = batch_size,
        classes = str(classes),
        data_augmentation = str(data_augmentations + [grayscale]),
        loss = loss.__name__,
        start_lr = start_lr,
        optimizer = get_module_name(optimizer),
        scheduler = get_module_name(scheduler),
        iou_threshold = iou_threshold,
        total_epochs = total_epochs,
        training_loss = losses['train'][-1],
        validation_loss = losses['val'][-1],
        train_iou_overall = metric_values['train']['iou_score_overall'][-1],
        val_iou_overall = metric_values['val']['iou_score_overall'][-1],
        train_iou_overall_best = metric_values['train']['iou_score_overall'][best_epoch],
        val_iou_overall_best = metric_values['val']['iou_score_overall'][best_epoch]
    )

    for _class in classes:
        result[f'train_iou_score_{_class}'] = metric_values['train'][f'iou_score_{_class}'][-1]
        result[f'val_iou_score_{_class}'] = metric_values['val'][f'iou_score_{_class}'][-1]
        result[f'val_iou_score_{_class}_best'] = metric_values['val'][f'iou_score_{_class}'][best_epoch]
    
    upload_google_sheets(result, logger = logging)