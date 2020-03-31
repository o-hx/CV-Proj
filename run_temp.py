import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
import torchvision

from utils.data import prepare_dataloader

# Boilerplate code for testing only. Uses SMP's high level APIs

if __name__ == '__main__':

    # Return relevant dataloaders from whatever matthew's function is
    cwd = os.getcwd()
    train_image_filepath = os.path.join(cwd,'data','train_images')
    test_image_filepath = os.path.join(cwd,'data','test_images')
    df_filepath = os.path.join(cwd,'data','train.csv')
    seed = 2

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataloader, validation_dataloader, test_dataloader = prepare_dataloader(train_image_filepath, test_image_filepath, df_filepath, seed, train_transform, test_transform)

    # Define Model
    segmentation_model = smp.Unet('resnet18', encoder_weights='imagenet',classes=4, activation='sigmoid')

    params = dict(
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose = True,
    )

    # Define Loss and Accuracy Metric
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params= segmentation_model.parameters(), lr=0.001),
    ])

    # Define Epochs
    train_epoch = smp.utils.train.TrainEpoch(
        segmentation_model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        **params
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        segmentation_model, 
        loss=loss, 
        metrics=metrics, 
        **params
    )

    # train model for 40 epochs
    max_score = 0

    for i in range(0, 5):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(validation_dataloader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(segmentation_model, './best_model.pth')
            print('Model saved!')
            
        if i == 25: # Note: Scheduler not implemented in SNP
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


    




