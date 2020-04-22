import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader, get_augmentations
from utils.train import test_model
from utils.misc import upload_google_sheets, get_module_name, log_print
from models.auxillary import BinaryFocalLoss
from models.unet import Unet

def mask2rle(pred):
    shape = pred.shape
    mask = np.zeros([shape[0], shape[1]], dtype=np.uint8)
    points = np.where(pred == 1)
    if len(points[0]) > 0:
        mask[points[0], points[1]] = 1
        mask = mask.reshape(-1, order='F')
        pixels = np.concatenate([[0], mask, [0]])
        rle = np.where(pixels[1:] != pixels[:-1])[0]
        rle[1::2] -= rle[::2]
    else:
        return ''
    return ' '.join(str(r) for r in rle)

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
    start_lr = 0.0005
    classes = ['gravel']
    iou_threshold = 0.5
    total_epochs = 20
    grayscale = False
    drop_empty = True
    loss_args = dict(
        beta = 0.8,
        gamma = 2.
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
                                                                                drop_empty = drop_empty
                                                                                )

    # Define Model
    segmentation_model = torch.load(os.path.join(os.getcwd(),'weights','Unet_EfficientNetEncoder_best_model.pth'))
    
    all_outputs, filepaths = test_model(test_dataloader,
                segmentation_model,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                predictions_save_path = os.path.join(os.getcwd(),'predictions'),
                )

    print(all_outputs.shape)
    print(filepaths)
    print(type(filepaths))

    