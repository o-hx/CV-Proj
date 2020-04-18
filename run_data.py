import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils.data import rle_to_mask


def images_in_class(df):
    # get count for each class
    fish = df[df['label'] == 'Fish'].EncodedPixels.count()
    flower = df[df['label'] == 'Flower'].EncodedPixels.count()
    gravel = df[df['label'] == 'Gravel'].EncodedPixels.count()
    sugar = df[df['label'] == 'Sugar'].EncodedPixels.count()

    # plot count against class label
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    count = [fish, flower, gravel, sugar]
    ax.bar(labels, count, color=['blue', 'orange', 'green', 'red'])
    plt.title('Number of images per class')
    fig.savefig('images_in_class.png')
    return fish, flower, gravel, sugar

def classes_per_image(df):
    # group by number of classes and get group count
    count = df.groupby('image')['EncodedPixels'].count()
    added = count.value_counts()
    
    # plot number of images against number of classes
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.hist(count, bins=4, edgecolor='white', linewidth=1)
    plt.title('Number of classes per image')
    fig.savefig('classes_per_image.png')
    return added
    
def get_binary_mask_sum(encoded_mask):
    mask_decoded = rle_to_mask(encoded_mask, width=2100, height=1400)
    binary_mask = (mask_decoded > 0.0).astype(int)
    return binary_mask.sum()

def get_mask_pixel_sum(df):
    df = df[df['EncodedPixels'].notna()]
    df['mask_pixel_sum'] = df.apply(lambda x: get_binary_mask_sum(x['EncodedPixels']), axis=1)
    return df

def mask_pixel_histogram(df):
    fish = df[df['label'] == 'Fish']['mask_pixel_sum']
    flower = df[df['label'] == 'Flower']['mask_pixel_sum']
    gravel = df[df['label'] == 'Gravel']['mask_pixel_sum']
    sugar = df[df['label'] == 'Sugar']['mask_pixel_sum']
    classes = ['fish', 'flower', 'gravel', 'sugar']
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0,0,1,1])
    ax.hist([fish,flower,gravel,sugar], bins=30, label=classes)
    ax.legend(prop={'size': 10})
    plt.title('Number of images against number of pixels in mask')
    fig.savefig('histogram.png')

    fig, axs = plt.subplots(2, 2, figsize=(15,15))
    axs[0, 0].hist(fish, bins=20)
    axs[0, 0].set_title('Fish')
    axs[0, 1].hist(flower, bins=20, color='orange')
    axs[0, 1].set_title('Flower')
    axs[1, 0].hist(gravel, bins=20, color='green')
    axs[1, 0].set_title('Gravel')
    axs[1, 1].hist(sugar, bins=20, color='red')
    axs[1, 1].set_title('Sugar')
    fig.savefig('histogram_separate.png')
    
    pass   



if __name__ == '__main__':
    cwd = os.getcwd()
    df_filepath = os.path.join(cwd,'data','train.csv')

    df = pd.read_csv(df_filepath)
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    
    images_in_class(df)
    print('images_in_class done')
    classes_per_image(df)
    print('classes_per_image done')

    print('calculating sum...')
    df_with_sum = get_mask_pixel_sum(df)
    print('done')

    mask_pixel_histogram(df_with_sum)
    print('mask_pixel_histogram done')


