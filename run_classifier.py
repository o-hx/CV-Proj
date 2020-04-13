import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data_classifier import prep_classification_data, get_augmentations
from utils.train import train_model
from utils.misc import upload_google_sheets, get_module_name, log_print
from models import BinaryFocalLoss, Accuracy

if __name__ == '__main__':
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    start_time = time.ctime()
    logging.basicConfig(filename= os.path.join(log_file_path,"training_log_classifier_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
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
    start_lr = 0.001
    classes = ['sugar','flower','fish','gravel']
    threshold = 0.5
    total_epochs = 10
    loss_args = dict(
        gamma = 2.
    )

    data_augmentation = get_augmentations(img_size)

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dl, valid_dl, test_dl = prep_classification_data(train_image_filepath = train_image_filepath,
                                                           df_filepath = df_filepath, 
                                                           seed = seed,
                                                           size = img_size, 
                                                           transforms = transforms,
                                                           data_augmentation = data_augmentation,
                                                           batch_size = batch_size)

    # Define Model
    classification_model = torchvision.models.densenet169(pretrained=True)
    classification_model.classifier = torch.nn.Sequential(torch.nn.Linear(in_features = 1664, out_features = 4, bias = True), torch.nn.Sigmoid())
    model_save_prefix = 'classifier_' + get_module_name(classification_model)

    params = dict(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose = True,
    )

    # Define Loss and Accuracy Metric
    loss = BinaryFocalLoss(gamma = loss_args['gamma']) #
    metrics = [
        Accuracy(threshold=threshold)
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params= classification_model.parameters(), lr=start_lr),
    ])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size =1, gamma=0.8)

    losses, metric_values, best_epoch, confusion_matrices = train_model(train_dataloader = train_dl,
                                                            validation_dataloader_list = [valid_dl],
                                                            model = classification_model,
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
        model_name = get_module_name(classification_model),
        image_size = str(img_size),
        batch_size = batch_size,
        classes = str(classes),
        data_augmentation = str(data_augmentation.transforms.transforms).replace('\n',''),
        loss = loss.__name__,
        loss_args = str(loss_args),
        start_lr = start_lr,
        optimizer = get_module_name(optimizer),
        scheduler = get_module_name(scheduler),
        acc_threshold = threshold,
        total_epochs = total_epochs,
        training_loss = losses['train'][-1],
        validation_loss = losses['val'][0][-1],
        train_acc_overall = metric_values['train']['accuracy_overall'][-1],
        val_acc_overall = metric_values['val'][0]['accuracy_overall'][-1],
        train_acc_overall_best = metric_values['train']['accuracy_overall'][best_epoch],
        val_acc_overall_best = metric_values['val'][0]['accuracy_overall'][best_epoch],
        train_cm = str(confusion_matrices['train']),
        val_cm = str(confusion_matrices['val'][0]),
    )

    for _class in classes:
        result[f'train_acc_{_class}'] = metric_values['train'][f'accuracy_{_class}'][-1]
        result[f'val_acc_{_class}'] = metric_values['val'][0][f'accuracy_{_class}'][-1]
        result[f'val_acc_{_class}_best'] = metric_values['val'][0][f'accuracy_{_class}'][best_epoch]
    
    upload_google_sheets(result, sheet_name = 'Classifier', logger = logging)