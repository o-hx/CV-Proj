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

def rle_to_mask(rle_string, width, height):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype= bool)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img

class Dataset(data.Dataset):
    def __init__(self, image_filepath, EncodedPixels, size, transforms = None, test = False, equalise = True, label = ['fish', 'flower', 'gravel', 'sugar'], data_augmentations = None, grayscale = False):
        self.image_filepath = image_filepath
        self.EncodedPixels = EncodedPixels # Each encodedpixels should be (4, H, W) for 4 masks, in the order of Fish, Flower, Gravel and Sugar masks
        self.transforms = transforms
        self.test = test
        self.size = size
        self.equalise = equalise
        self.grayscale = grayscale
        self.label = label
        self.list_of_classes = ['fish', 'flower', 'gravel', 'sugar']
        self.data_augmentations = data_augmentations

    def __len__(self):
        return len(self.image_filepath)
    def __getitem__(self, index):
        ID = self.image_filepath[index]
        # Load data and get label
        X = histogram_equalize(ID, self.equalise, self.grayscale)
        if self.transforms is not None:
            X = self.transforms(X)
        if self.test:
            return X
        else:
            masks = self.EncodedPixels[index]
            masks = [torch.from_numpy((rle_to_mask(i, 2100, 1400) * 1).astype(float)).unsqueeze(0) for i in masks]
            masks = torch.stack(masks)
            masks = torch.nn.functional.interpolate(masks, size = self.size).squeeze().type(torch.float32)
            trans = torchvision.transforms.ToPILImage()
            trans1 = torchvision.transforms.ToTensor()
            if self.data_augmentations is not None:
                for t in self.data_augmentations:
                    if np.random.random_sample() >= 0.5:
                        X = trans(X)
                        X = t(X)
                        X = trans1(X)
                        mask_stack = []
                        for i in range(len(self.list_of_classes)):
                            temp_mask = trans(masks[i])
                            temp_mask = t(temp_mask)
                            temp_mask = trans1(temp_mask)
                            mask_stack.append(temp_mask)
                        masks = torch.stack(mask_stack)
                        masks.squeeze_()

            for lab in self.label:
                if lab not in self.list_of_classes:
                    raise ValueError("Make sure all labels belong to 'fish', 'flower', 'gravel' or 'sugar'")
            masks_to_include = [masks[self.list_of_classes.index(lab), :,:] for lab in self.label]
            masks = torch.stack(masks_to_include)
            return X, masks

def histogram_equalize(filepath, equalise = False, grayscale = False):
    if grayscale:
        img = cv2.imread(filepath, 0)
        ret, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
        img = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.imread(filepath, 1)

    if equalise:
         # read a image using imread 
        b,g,r = cv2.split(img)
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))
        img = Image.fromarray(equ)
    else:
        img = Image.fromarray(img)
    return img

def split_data(df_filepath, seed, train_proportion = 0.9):
    df = pd.read_csv(df_filepath)
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df['EncodedPixels'] = df['EncodedPixels'].fillna(-1)
    list_of_images = df['image'].unique()
    random.seed(seed)    
    random.shuffle(list_of_images)
    train_df = df[df['image'].isin(list_of_images[:int(len(list_of_images) * train_proportion)])]
    valid_df = df[df['image'].isin(list_of_images[int(len(list_of_images) * train_proportion):])]
    return train_df, valid_df

def group_data(df, image_filepath):
    images = df['image'].tolist()

    images = [image_filepath + '/' + images[i] for i in range(0, len(images), 4)]
    encodedpixels = df['EncodedPixels'].tolist()
    masks = [encodedpixels[i:i+4] for i in range(0, len(encodedpixels), 4)]
    return images, masks

def prepare_dataloader(train_image_filepath, test_image_filepath, df_filepath, seed, train_transform, test_transform, size, batch_size = 1, shuffle_val_dataloader = False, label = None, data_augmentations = None, grayscale = False):
    train_df, valid_df = split_data(df_filepath, seed)
    train_images, train_masks = group_data(train_df, train_image_filepath)
    valid_images, valid_masks = group_data(valid_df, train_image_filepath)
    test_images = [test_image_filepath + '/' + i for i in os.listdir(test_image_filepath)]
    train_ds = Dataset(train_images, train_masks, size = size, transforms = train_transform, label = label, data_augmentations = data_augmentations, grayscale = grayscale)
    valid_ds = Dataset(valid_images, valid_masks, size = size, transforms = test_transform, label = label, grayscale = grayscale)
    test_ds = Dataset(test_images, EncodedPixels = None, transforms = test_transform,  size = size, test = True, label = label, grayscale = grayscale)

    train_dl = data.DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    valid_dl = data.DataLoader(valid_ds, batch_size = batch_size, shuffle = shuffle_val_dataloader)
    test_dl = data.DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    return train_dl, valid_dl, test_dl


if __name__ == "__main__":

    cwd = os.getcwd()
    train_image_filepath = f'{cwd}/train_images'
    test_image_filepath = f'{cwd}/test_images'
    df_filepath = f'{cwd}/train.csv'
    seed = 2

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_augmentations = [torchvision.transforms.RandomHorizontalFlip(p= 1), 
                          torchvision.transforms.RandomVerticalFlip(p= 1)]

    train_dl, valid_dl, test_dl = prepare_dataloader(train_image_filepath, test_image_filepath, df_filepath, seed, train_transform, test_transform, 256, 64, label = ['fish'], data_augmentations = data_augmentations, grayscale = True)