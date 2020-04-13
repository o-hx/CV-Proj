import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
import os
import torch
import torchvision
import cv2
import random
import torch.functional as F
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    Cutout,
    Normalize,
    Compose,
    GaussNoise,
    IAAAdditiveGaussianNoise,
    RandomContrast,
    RandomGamma,
    RandomRotate90,
    RandomSizedCrop,
    RandomBrightness,
    Resize,
    ShiftScaleRotate,
    MotionBlur,
    MedianBlur,
    Blur,
    OpticalDistortion,
    GridDistortion,
    IAAPiecewiseAffine,
    OneOf)

def get_augmentations(img_size):
    height, width = img_size
    list_transforms = []
    list_transforms.append(HorizontalFlip())
    list_transforms.append(VerticalFlip())
    list_transforms.append(
        Resize(
            height = int(height * 1.5),
            width = int(width * 1.5)
            )
    )
    list_transforms.append(
        RandomSizedCrop(
            min_max_height=(int(height * 0.90), height),
            height=height,
            width=width,
            w2h_ratio=width/height)
    )

    list_transforms.append(
        OneOf([
            GaussNoise(),
            IAAAdditiveGaussianNoise(),
        ], p=0.5),
    )
    list_transforms.append(
        OneOf([
            RandomContrast(0.5),
            RandomGamma(),
            RandomBrightness(),
        ], p=0.5),
    )
    num_holes = 10
    hole_size = 25
    list_transforms.append(Cutout(num_holes, hole_size))
    return Compose(list_transforms)

def histogram_equalize(filepath, equalise = False):
    img = cv2.imread(filepath, 1)
    if equalise:
         # read a image using imread 
        b,g,r = cv2.split(img)
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        img = cv2.merge(( equ_r,  equ_g, equ_b))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def split_data(df_filepath, seed, train_proportion = 0.9):
    classes = ['fish', 'flower', 'gravel', 'sugar']
    df = pd.read_csv(df_filepath)
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df['EncodedPixels'] = df['EncodedPixels'].fillna(0)
    df['EncodedPixels'] = np.where(df['EncodedPixels'] != 0, 1, 0)

    list_of_images = df['image'].unique()
    random.seed(seed)    
    random.shuffle(list_of_images)

    test_proportion = (1 - train_proportion)/2

    train_set = list_of_images[:int(len(list_of_images) * train_proportion)]
    valid_set = list_of_images[int(len(list_of_images) * train_proportion):int(len(list_of_images) * (train_proportion + test_proportion))]
    test_set = list_of_images[int(len(list_of_images) * (train_proportion + test_proportion)):]

    train_dict = {}
    for img in train_set:
        train_dict[img] = df[df['image'] == img]['EncodedPixels'].tolist()

    valid_dict = {}
    for val_img in valid_set:
        valid_dict[val_img] = df[df['image'] == val_img]['EncodedPixels'].tolist() 

    test_dict = {}
    for val_img in test_set:
        test_dict[val_img] = df[df['image'] == val_img]['EncodedPixels'].tolist() 
    return train_dict, valid_dict, test_dict, classes

class classification_Dataset(data.Dataset):
    def __init__(self, train_image_filepath, image_filepath, image_labels, size, transforms = None, data_augmentation = None, equalise = True, list_of_classes = None):
        self.train_image_filepath = train_image_filepath
        self.image_filepath = image_filepath
        self.image_labels = image_labels
        self.transforms = transforms
        self.size = size
        self.equalise = equalise
        self.list_of_classes = list_of_classes
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.image_filepath)
    def __getitem__(self, index):
        ID = self.image_filepath[index]
        # Load data and get label
        X = histogram_equalize(ID, self.equalise)
        if self.data_augmentation is not None:
            data = {"image": X}
            augmented = self.data_augmentation(**data)
            X = augmented['image']
        if self.transforms is not None:
            X = Image.fromarray(X)
            X = self.transforms(X)
        label = torch.FloatTensor(self.image_labels[ID.replace(self.train_image_filepath + '/', '')])
        if len(self.list_of_classes) == 1 and self.list_of_classes is not None:
            idx = ['fish', 'flower', 'gravel', 'sugar'].index(self.list_of_classes[0])
            label = label[idx]
        return X, label

def prep_classification_data(train_image_filepath,
                             df_filepath, 
                             seed,
                             size, 
                             transforms,
                             data_augmentation,
                             batch_size,
                             equalise = True,
                             train_proportion = 0.9, 
                             list_of_classes = ['sugar','flower','fish','gravel']):

    # Get dictionary of {image: labels}
    train_dict, valid_dict, test_dict, classes = split_data(df_filepath, seed, train_proportion = train_proportion)

    # Create list of image filepaths
    train_fp = [train_image_filepath + '/' + img for img in train_dict.keys()]
    valid_fp = [train_image_filepath + '/' + img for img in valid_dict.keys()]
    test_fp = [train_image_filepath + '/' + img for img in test_dict.keys()]


    # Initialise classification dataset class 
    train_ds = classification_Dataset(train_image_filepath, train_fp, train_dict, size = size, transforms = transforms, data_augmentation = data_augmentation, equalise = equalise, list_of_classes = list_of_classes)
    valid_ds = classification_Dataset(train_image_filepath, valid_fp, valid_dict, size = size, transforms = transforms, data_augmentation = None, equalise = equalise, list_of_classes = list_of_classes)
    test_ds = classification_Dataset(train_image_filepath, test_fp, test_dict, size = size, transforms = transforms, data_augmentation = None, equalise = equalise, list_of_classes = list_of_classes)
    
    # Initialise dataloader
    train_dl = data.DataLoader(train_ds, batch_size = batch_size, num_workers=batch_size)
    valid_dl = data.DataLoader(valid_ds, batch_size = batch_size, num_workers=batch_size)
    test_dl = data.DataLoader(test_ds, batch_size = batch_size, num_workers=batch_size)

    return train_dl, valid_dl, test_dl

if __name__ == "__main__":

    cwd = os.getcwd()
    train_image_filepath = f'{cwd}/data/train_images'
    df_filepath = f'{cwd}/data/train.csv'
    seed = 3
    img_size = (6*64, 9*64)
    train_proportion = 0.9
    equalise = True
    batch_size = 2
    dl_class = ['fish'] # This should be a single class, or None if want to include all
    data_augmentation = get_augmentations(img_size)
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dl, valid_dl, test_dl = prep_classification_data(train_image_filepath = train_image_filepath,
                                                           df_filepath = df_filepath, 
                                                           seed = seed, 
                                                           train_proportion = train_proportion, 
                                                           size = img_size, 
                                                           transforms = transforms,
                                                           data_augmentation = data_augmentation,
                                                           equalise = equalise, 
                                                           batch_size = batch_size, 
                                                           list_of_classes = dl_class)

    for idx, X in enumerate(train_dl):
        x, label = X
        print(x.shape)
        print(label.shape)
        print(label[0])
        trans = torchvision.transforms.ToPILImage()
        img = np.array(trans(x[0]))
        plt.imshow(img)
        plt.show()
        break