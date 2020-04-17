import numpy as np
import cv2
import os
import pandas as pd
from torch.utils import data
import glob
from torchvision import transforms
from PIL import Image


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

def rle2bb(rle):
    if rle=='': return (0,0,0,0)
    mask = rle2mask(rle)
    z = np.argwhere(mask==1)
    mn_x = np.min( z[:,0] )
    mx_x = np.max( z[:,0] )
    mn_y = np.min( z[:,1] )
    mx_y = np.max( z[:,1] )
    return (mn_x,mn_y,mx_x-mn_x,mx_y-mn_y)

def prep_df(df_filepath):
    df = pd.read_csv(df_filepath)
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df = df.dropna()
    return df

class Dataset(data.Dataset):
    def __init__(self, image_filepath, EncodedPixels, size, label):
        self.image_filepath = image_filepath
        self.EncodedPixels = EncodedPixels
        self.size = size
        self.label = label
    def __len__(self):
        return len(self.image_filepath)
    def __getitem__(self, index):
        ID = self.image_filepath[index]
        x1, y1, x2, y2 = rle2bb(self.EncodedPixels[index])
        class_label = self.label[index]
        img = Image.open(ID)
        img = img.crop((x1, y1, x2, y2))
        resize = transforms.Resize(self.size)
        trans1 = transforms.ToTensor()
        img = trans1(resize(img))
        return img, self.label


def get_dataloader(df_filepath, train_image_filepath, img_size):
    df = prep_df(df_filepath)
    image_filepath = [f'{train_image_filepath}/{i}' for i in df['image'].tolist()]
    labels = df['label'].tolist()
    encodedpixels = df['EncodedPixels'].tolist()

    assert len(image_filepath) == len(labels) == len(encodedpixels), "Make sure lengths same"

    dl = data.DataLoader(Dataset(image_filepath, encodedpixels, img_size, labels))
    return dl

if __name__ == "__main__":
    cwd = os.getcwd()
    train_image_filepath = r'C:\Users\Dell\Desktop\CV-Proj\data\train_images'    
    df_filepath = r'C:\Users\Dell\Desktop\CV-Proj\data\train.csv'
    seed = 2
    img_size = (250, 250)
    dl = get_dataloader(df_filepath, train_image_filepath, img_size)
    for idx, (img, label) in enumerate(dl):
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        trans(img[0]).show()
        break