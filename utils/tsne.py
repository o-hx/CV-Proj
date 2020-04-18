import numpy as np
import cv2
import os
import pandas as pd
from torch.utils import data
import glob
from torchvision import transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import matplotlib as mpl
from itertools import cycle

def rle_to_mask(rle_string, width = 2100, height = 1400):
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

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin, cmin, rmax, cmax)

def prep_df(df_filepath, label):
    df = pd.read_csv(df_filepath)
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df = df[df['label'] == label.capitalize()]
    df = df.dropna()
    return df
class IterDataset(data.IterableDataset):
    def __init__(self, image_filepath, EncodedPixels, size, normalise):
        self.image_filepath = image_filepath
        self.EncodedPixels = EncodedPixels
        self.size = size
        self.normalise = normalise
    def get_masks(self):
        for idx in range(len(self.image_filepath)):
            ID = self.image_filepath[idx]
            mask = np.array((rle_to_mask(self.EncodedPixels[idx], 2100, 1400) * 1)).T
            img = Image.open(ID)
            mask = np.where(mask > 0, 1, 0)
            lbl = label(mask)
            props = regionprops(lbl)
            props_list = [prop for prop in props if prop.area > 60000]
            
            if sum([(prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1]) for prop in props_list]) == 0:
                continue
            iou = (sum([prop.area for prop in props_list]))/ (sum([(prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1]) for prop in props_list]))
            if iou < 0.8:
                continue
            else:
                for props in props_list:
                    img_cropped = img.crop((props.bbox[0], props.bbox[1], props.bbox[2], props.bbox[3]))
                    trans1 = transforms.ToTensor()
                    resize = transforms.Resize(self.size)
                    img_cropped = trans1(resize(img_cropped))
                    if self.normalise:
                        norma = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        img_cropped = norma(img_cropped)
                    yield img_cropped, img_cropped
    def __iter__(self):        
        return self.get_masks()

def get_dataloader(df_filepath, train_image_filepath, img_size, label, normalise, batch_size):
    assert label in ['fish', 'sugar', 'gravel', 'flower'], "Please choose one of the following: 'fish', 'sugar', 'gravel', 'flower'"
    df = prep_df(df_filepath, label)
    image_filepath = [f'{train_image_filepath}/{i}' for i in df['image'].tolist()]
    encodedpixels = df['EncodedPixels'].tolist()
    assert len(image_filepath) == len(encodedpixels), "Make sure lengths same"

    # Split train & val
    image_filepath_trainset = image_filepath[:int(len(image_filepath) * 0.8)]
    image_filepath_valset = image_filepath[int(len(image_filepath) * 0.8):]

    encodedpixels_trainset = encodedpixels[:int(len(encodedpixels) * 0.8)]
    encodedpixels_valset = encodedpixels[int(len(encodedpixels) * 0.8):]

    assert len(image_filepath_trainset) == len(encodedpixels_trainset), f"Check length of image_filepath_trainset: {len(image_filepath_trainset)} and encodedpixels_trainset: {len(encodedpixels_trainset)}"
    assert len(image_filepath_valset) == len(encodedpixels_valset), f"Check length of image_filepath_valset: {len(image_filepath_valset)} and encodedpixels_valset: {len(encodedpixels_valset)}"

    train_dl = data.DataLoader(IterDataset(image_filepath_trainset, encodedpixels_trainset, img_size, normalise), batch_size=batch_size, drop_last = True, num_workers=12)
    val_dl = data.DataLoader(IterDataset(image_filepath_valset, encodedpixels_valset, img_size, normalise), batch_size=batch_size, drop_last = True, num_workers=8)

    return train_dl, val_dl

if __name__ == "__main__":
    cwd = os.getcwd()
    train_image_filepath = r'C:\Users\Dell\Desktop\CV-Proj\data\train_images'    
    df_filepath = r'C:\Users\Dell\Desktop\CV-Proj\data\train.csv'
    lab = 'flower'
    img_size = (250, 250)
    normalise = True
    batch_size = 8
    train_dl, val_dl = get_dataloader(df_filepath, train_image_filepath, img_size, lab, normalise, batch_size)
    tot = 0
    for idx1, img in enumerate(train_dl):
        tot += img.shape[0]
    if tot % batch_size * 100:
        print(tot)
    print(tot)