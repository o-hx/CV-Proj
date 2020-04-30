import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision
import pandas as pd

from utils.data import prepare_dataloader, get_augmentations
from utils.train import test_model
from utils.misc import upload_google_sheets, get_module_name, log_print
from models.auxillary import BinaryFocalLoss
from models.unet import Unet
from utils.data_classifier import prep_classification_data
from models.enhanced import CloudSegment

def mask2rle(pred):
    shape = pred.shape
    mask = np.zeros([shape[0], shape[1]], dtype=np.uint8)
    pred = np.where(pred > 0.5, 1, 0)
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

    # Classification
    cwd = os.getcwd()
    train_image_filepath = os.path.join(cwd,'data','train_images')
    test_image_filepath = os.path.join(cwd,'data','test_images')
    df_filepath = os.path.join(cwd,'data','train.csv')
    batch_size = 1
    seed = 2
    classification_img_size = (10*64, 15*64)
    actual_img_size = (1400, 2100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classification_model = os.path.join(os.getcwd(),'weights','classifier.pth')
    segmentation_model_fish  = os.path.join(os.getcwd(),'weights','final_fishUnet_EfficientNetEncoder_current_model.pth')
    segmentation_model_flower = os.path.join(os.getcwd(),'weights','final_flowerUnet_EfficientNetEncoder_current_model.pth')
    segmentation_model_gravel = os.path.join(os.getcwd(),'weights','final_gravelUnet_EfficientNetEncoder_current_model.pth')
    segmentation_model_sugar = os.path.join(os.getcwd(),'weights','final_sugarUnet_EfficientNetEncoder_current_model.pth')
    
    model = CloudSegment(classification_model,
        sugar_path = segmentation_model_sugar,
        flower_path = segmentation_model_flower,
        fish_path = segmentation_model_fish,
        gravel_path = segmentation_model_gravel,
        classifier_class_order = ['flower', 'gravel', 'sugar', 'fish'],
        classifier_threshold = 0.5,
        dataloader_class_order = ['fish','flower','gravel','sugar'],
        device = device)

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(classification_img_size),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    _, _, classifier_test_dl, _ = prep_classification_data(train_image_filepath = train_image_filepath,
                                                           test_image_filepath = test_image_filepath,
                                                           df_filepath = df_filepath, 
                                                           seed = seed,
                                                           size = classification_img_size,
                                                           list_of_classes = ['flower', 'gravel', 'sugar', 'fish'],
                                                           transforms = transforms,
                                                           data_augmentation = None,
                                                           batch_size = batch_size)

    upsample_transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), 
                                                         torchvision.transforms.Resize(actual_img_size),
                                                         torchvision.transforms.ToTensor()])
    
    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    if torch.cuda.is_available():
        model.cuda()

    all_outputs = []
    prediction_filepath = []

    with torch.no_grad():
        for _, data in enumerate(classifier_test_dl):
            inputs = data[0].to(device)
            for fp in data[1]:
                for lab  in ['fish','flower','gravel','sugar']:
                    prediction_filepath += [fp + '_' + lab.capitalize()]
            outputs = model(inputs)
            masks, labels = outputs
            for batch in masks:
                for clas in batch:
                    clas = upsample_transform(clas).cpu().detach().numpy()
                    all_outputs.append(mask2rle(clas.squeeze()))

    submission_dict = {'Image_Label': prediction_filepath, 'EncodedPixels': all_outputs}
    submission = pd.DataFrame.from_dict(submission_dict)
    submission.to_csv('submission.csv', index=False)