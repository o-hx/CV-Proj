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
from utils.data_classifier import prep_classification_data

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
    classification_img_size = (int(10*64), int(15*64))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(classification_img_size),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    model_class_order = ['flower', 'gravel', 'sugar', 'fish']
    submission_order = ['fish','flower', 'gravel', 'sugar']
    reorder_index = [model_class_order.index(submission_label) for submission_label in submission_order]
    
    _, _, classifier_test_dl, _ = prep_classification_data(train_image_filepath = train_image_filepath,
                                                           test_image_filepath = test_image_filepath,
                                                           df_filepath = df_filepath, 
                                                           seed = seed,
                                                           size = classification_img_size,
                                                           list_of_classes = model_class_order,
                                                           transforms = transforms,
                                                           data_augmentation = None,
                                                           batch_size = batch_size)
    classification_model = torch.load(os.path.join(os.getcwd(),'weights','flower_gravel_sugar_fishclassifier_EfficientNetbest_model.pth'))
    if not os.path.exists('predictions'):
        os.mkdir('predictions')
    if torch.cuda.is_available():
        classification_model.cuda()

    all_classification_outputs = []
    classification_filepath = []

    with torch.no_grad():
        for _, data in enumerate(classifier_test_dl):
            inputs = data[0].to(device)
            classification_filepath += [i for i in data[1]]
            outputs = classification_model(inputs)
            outputs = torch.index_select(outputs, 1, torch.LongTensor(reorder_index).to(device))
            outputs = outputs.cpu().detach().numpy()
            all_classification_outputs.append(np.where(outputs > 0.5, 1, 0))
        all_classification_outputs = np.concatenate(all_classification_outputs, axis = 0)

    segmentation_model_fish = torch.load(os.path.join(os.getcwd(),'weights','final_fishUnet_EfficientNetEncoder_current_model.pth'))
    segmentation_model_flower = torch.load(os.path.join(os.getcwd(),'weights','final_flowerUnet_EfficientNetEncoder_current_model.pth'))
    segmentation_model_gravel = torch.load(os.path.join(os.getcwd(),'weights','final_gravelUnet_EfficientNetEncoder_current_model.pth'))
    segmentation_model_sugar = torch.load(os.path.join(os.getcwd(),'weights','final_sugarUnet_EfficientNetEncoder_current_model.pth'))
    
    fish_segmentation_outputs = []
    flower_segmentation_outputs = []
    gravel_segmentation_outputs = []
    sugar_segmentation_outputs = []
    segmentation_filepath = []
    
    with torch.no_grad():
        for _, data in enumerate(classifier_test_dl):
            inputs = data[0].to(device)
            segmentation_filepath += [i for i in data[1]]
            outputs = segmentation_model_fish(inputs)
            fish_segmentation_outputs += [mask2rle(mask) for mask in outputs.cpu().detach().numpy()]
            print(fish_segmentation_outputs)
            outputs = segmentation_model_flower(inputs)
            flower_segmentation_outputs += [mask2rle(mask) for mask in outputs.cpu().detach().numpy()]
            outputs = segmentation_model_gravel(inputs)
            gravel_segmentation_outputs += [mask2rle(mask) for mask in outputs.cpu().detach().numpy()]
            outputs = segmentation_model_sugar(inputs)
            sugar_segmentation_outputs += [mask2rle(mask) for mask in outputs.cpu().detach().numpy()]