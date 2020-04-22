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
    def __init__(self, image_filepath, EncodedPixels, size, transforms = None, mask_transform = None, test = False, equalise = True, label = ['fish', 'flower', 'gravel', 'sugar'], data_augmentations = None, grayscale = False):
        self.image_filepath = image_filepath
        self.EncodedPixels = EncodedPixels # Each encodedpixels should be (4, H, W) for 4 masks, in the order of Fish, Flower, Gravel and Sugar masks
        self.transforms = transforms
        self.mask_transform = mask_transform
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
        if self.test and self.transforms is not None:
            X = Image.fromarray(X)
            X = self.transforms(X)
            return X, ID
        else:
            masks = self.EncodedPixels[index]
            masks = np.array([(rle_to_mask(i, 2100, 1400) * 1).astype(float) for i in masks])

            if self.data_augmentations is not None:
                data = {"image": X, "masks": masks}
                augmented = self.data_augmentations(**data)
                X, masks = augmented['image'], augmented['masks']
            if self.transforms is not None:
                X = Image.fromarray(X)
                X = self.transforms(X)
            if self.mask_transform is not None:
                masks = [Image.fromarray(np.array(mask)) for mask in masks]
                masks = [self.mask_transform(mask) for mask in masks]
                masks = torch.stack(masks)      
            for lab in self.label: # Only includes the masks provided in self.label
                if lab not in self.list_of_classes:
                    raise ValueError("Make sure all labels belong to 'fish', 'flower', 'gravel' or 'sugar'")
            masks_to_include = [masks[self.list_of_classes.index(lab), :,:] for lab in self.label]
            masks = torch.squeeze(torch.stack(masks_to_include), 1)
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
        img = cv2.merge(( equ_r,  equ_g, equ_b))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def drop_empty_df(df, label):
    list_of_all_empties = [set(df[(df['label'] == lab.title()) & (df['EncodedPixels'] == -1)]['image'].tolist())for lab in label]
    common_empty_images = list(set.intersection(*list_of_all_empties))
    df = df[~df['image'].isin(common_empty_images)]
    return df

def prepare_dataloader(train_image_filepath,
                        test_image_filepath,
                        df_filepath,
                        seed,
                        train_transform,
                        test_transform,
                        mask_transform,
                        size,
                        batch_size = 1,
                        shuffle_train_dataloader = True,
                        shuffle_val_dataloader = False,
                        label = None,
                        data_augmentations = None,
                        train_proportion = 0.9,
                        equalise = True,
                        grayscale = False,
                        drop_empty = True):

    train_df, valid_df = split_data(df_filepath, seed, train_proportion = train_proportion)

    # Create df that drops image if it does not have any masks in the label provided
    if len(label) < 4 and drop_empty:
        train_df_no_empty = drop_empty_df(train_df, label)
        valid_df_no_empty = drop_empty_df(valid_df, label)
    else:
        train_df_no_empty = train_df
        valid_df_no_empty = None

    train_images_no_empty, train_masks_no_empty = group_data(train_df_no_empty, train_image_filepath)
    valid_images, valid_masks = group_data(valid_df, train_image_filepath)
    if len(label) < 4 and drop_empty:
        valid_images_no_empty, valid_masks_no_empty = group_data(valid_df_no_empty, train_image_filepath)
    else:
        valid_images_no_empty, valid_masks_no_empty = None, None
    test_images = [test_image_filepath + '/' + i for i in os.listdir(test_image_filepath)][:100]

    train_ds = Dataset(train_images_no_empty, train_masks_no_empty, size = size, test = False, transforms = train_transform, mask_transform = mask_transform, label = label, data_augmentations = data_augmentations, grayscale = grayscale, equalise = equalise)
    valid_ds = Dataset(valid_images, valid_masks, size = size, test = False, transforms = test_transform, mask_transform = mask_transform, label = label, grayscale = grayscale, equalise = equalise)
    if len(label) < 4 and drop_empty:
        valid_ds_no_empty = Dataset(valid_images_no_empty, valid_masks_no_empty, size = size, test = False, transforms = test_transform, mask_transform = mask_transform, label = label, grayscale = grayscale, equalise = equalise)
    else:
        valid_ds_no_empty = None
    test_ds = Dataset(test_images, EncodedPixels = None, transforms = test_transform,  size = size, test = True, label = label, grayscale = grayscale, equalise = equalise)

    train_dl = data.DataLoader(train_ds, batch_size = batch_size, shuffle = shuffle_train_dataloader, num_workers=12)
    valid_dl = data.DataLoader(valid_ds, batch_size = batch_size, shuffle = shuffle_val_dataloader, num_workers=8)
    if len(label) < 4 and drop_empty:
        valid_dl_no_empty = data.DataLoader(valid_ds_no_empty, batch_size = batch_size, shuffle = shuffle_val_dataloader)
    else:
        valid_dl_no_empty = None
    test_dl = data.DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    return train_dl, valid_dl, valid_dl_no_empty, test_dl

if __name__ == "__main__":

    cwd = os.getcwd()
    train_image_filepath = f'{cwd}/data/train_images'
    test_image_filepath = f'{cwd}/data/test_images'
    df_filepath = f'{cwd}/data/train.csv'
    seed = 2
    img_size = (6*64, 9*64)

    mask_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor()
                                                    ])

    train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ])

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((6*64, 9*64)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_augmentations = get_augmentations(img_size)

    train_dl, valid_dl, valid_dl_no_empty, test_dl = prepare_dataloader(train_image_filepath, test_image_filepath, df_filepath, seed, train_transform, test_transform, mask_transform, (6*64, 9*64), 16, label = ['fish'], data_augmentations = data_augmentations, grayscale = False, equalise = True, drop_empty = False)

    for idx, X in enumerate(train_dl):
        x, mask = X
        print(x.shape)
        print(mask.shape)
        trans = torchvision.transforms.ToPILImage()
        img = np.array(trans(x[0]))
        plt.imshow(img)
        plt.show()
        break

    # for idx, X in enumerate(valid_dl):
    #     x, mask = X
    #     print(x.shape)
    #     print(mask.shape)
    #     trans = torchvision.transforms.ToPILImage()
    #     print(x.shape)
    #     img = np.array(trans(x[0]))
    #     plt.imshow(img)
    #     plt.show()
    #     break

    # for idx, X in enumerate(valid_dl_no_empty):
    #     x, mask = X
    #     print(x.shape)
    #     print(mask.shape)
    #     trans = torchvision.transforms.ToPILImage()
    #     print(x.shape)
    #     img = np.array(trans(x[0]))
    #     plt.imshow(img)
    #     plt.show()
    #     break

    # for idx, X in enumerate(test_dl):
    #     x = X
    #     print(x.shape)
    #     trans = torchvision.transforms.ToPILImage()
    #     print(x.shape)
    #     img = np.array(trans(x[0]))
    #     plt.imshow(img)
    #     plt.show()
    #     break