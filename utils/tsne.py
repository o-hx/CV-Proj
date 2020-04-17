import numpy as np
import cv2
import os
import pandas as pd
from torch.utils import data
import glob
from torchvision import transforms
from PIL import Image

def rle2mask(mask_rle, shape=(2100,1400), shrink=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T[::shrink,::shrink]

def rle2bb(rle):
    if rle=='': return (0,0,0,0)
    mask = rle2mask(rle)
    z = np.argwhere(mask==1)
    mn_x = np.min( z[:,0] )
    mx_x = np.max( z[:,0] )
    mn_y = np.min( z[:,1] )
    mx_y = np.max( z[:,1] )
    return (mn_x,mn_y,mx_x-mn_x,mx_y-mn_y)

def prep_df(df_filepath, label):
    df = pd.read_csv(df_filepath)
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df = df[df['label'] == label.capitalize()]
    df = df.dropna()
    return df

class Dataset(data.Dataset):
    def __init__(self, image_filepath, EncodedPixels, size, normalise):
        self.image_filepath = image_filepath
        self.EncodedPixels = EncodedPixels
        self.size = size
        self.normalise = normalise

    def __len__(self):
        return len(self.image_filepath)

    def __getitem__(self, index):
        ID = self.image_filepath[index]
        x1, y1, x2, y2 = rle2bb(self.EncodedPixels[index])
        img = Image.open(ID)
        img = img.crop((x1, y1, x2, y2))
        resize = transforms.Resize(self.size)
        trans1 = transforms.ToTensor()
        img = trans1(resize(img))
        if normalise:
            norma = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = norma(img)
        return img


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
    encodedpixels_valset = encodedpixels[:int(len(encodedpixels) * 0.8)]

    train_dl = data.DataLoader(Dataset(image_filepath_trainset, encodedpixels_trainset, img_size, normalise), batch_size=batch_size)
    val_dl = data.DataLoader(Dataset(image_filepath_valset, encodedpixels_valset, img_size, normalise), batch_size=batch_size)

    return train_dl, val_dl

if __name__ == "__main__":
    cwd = os.getcwd()
    train_image_filepath = r'C:\Users\Dell\Desktop\CV-Proj\data\train_images'    
    df_filepath = r'C:\Users\Dell\Desktop\CV-Proj\data\train.csv'
    seed = 2
    label = 'flower'
    img_size = (250, 250)
    normalise = True
    batch_size = 16

    train_dl, val_dl = get_dataloader(df_filepath, train_image_filepath, img_size, label, normalise, batch_size)
    print(len(train_dl))
    print(len(val_dl))
    for idx, img in enumerate(val_dl):
        print(img.shape)
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        trans(img[0]).show()
        break