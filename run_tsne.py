import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data import prepare_dataloader, get_augmentations
from utils.train import train_model, cluster, plot_clusters
from utils.misc import upload_google_sheets, get_module_name, log_print
from utils.tsne import get_dataloader
from models.autoencoder import Autoencoder
from models.auxillary import MSE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # Set up logging
    log_file_path = os.path.join(os.getcwd(),'logs')
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    start_time = time.ctime()
    logging.basicConfig(filename= os.path.join(log_file_path,"autoencode_log_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".log"),
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    # Return relevant dataloaders from whatever matthew's function is
    cwd = os.getcwd()
    train_image_filepath = os.path.join(cwd,'data','train_images')
    df_filepath = os.path.join(cwd,'data','train.csv')
    seed = 2
    batch_size = 256
    img_size = (287, 287)
    start_lr = 0.001
    total_epochs = 10
    k = 5
    col_dict = {'flower': 'blue', 'fish' : 'green', 'gravel' : 'black', 'sugar' : 'pink'}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Intra-class t-sne
    for classes in ['flower', 'fish', 'sugar', 'gravel']:
        autoencoder = torch.load(os.path.join(os.getcwd(),'weights',f'{classes}_Autoencodercurrent_model.pth'), map_location = device)
        autoencoder.eval()
        with torch.no_grad():
            train_dataloader, validation_dataloader = get_dataloader(df_filepath = df_filepath,
                                                train_image_filepath = train_image_filepath,
                                                img_size = img_size,
                                                label = classes,
                                                normalise = True,
                                                batch_size = batch_size
                                                )
            for _, data in enumerate(train_dataloader):
                img, _ = data
                img = img.to(device)
                _ = autoencoder.forward(img)
                latent = autoencoder.latent.detach().cpu().numpy()
                break
            tsne_output_all_four = []
            col_labels = []
            perplexities = np.arange(5, 60, 10)
            fig = plt.figure(figsize=(10,10), dpi= 100)
            fig2 = plt.figure(figsize=(10,10))
            fig2,ax2 = plt.subplots(5,5, figsize = (25,25))
            fig.suptitle(classes)
            counter = 0
            for i in range(len(perplexities)):
                X_embedded = TSNE(n_components=2, perplexity = perplexities[i]).fit_transform(latent)
                print(f'TSNE built for {classes} intra-class, perplexity: {perplexities[i]}')
                x = X_embedded[:,0]
                y = X_embedded[:,1]
                print(type(x))
                outliers = np.where((x > np.percentile(x, 99)) | (x < np.percentile(x,1) | (y > np.percentile(x, 99) | (y < np.percentile(x,1)))))
                for idx in range(len(list(outliers[0]))):
                    counter += 1
                    if counter >= 25:
                        break
                    ax2[counter//5,counter % 5].imshow(img[list(outliers[0])[idx]].permute(1, 2, 0))
                    ax2.set_title(f'Outlier for perplexity {perplexities[i]}')                
                ax = fig.add_subplot(2, 3, i+1)
                ax.scatter(x, y, alpha = 0.3)
                ax.set_title(f'Perplexity: {perplexities[i]}')
            fig.savefig(f'intraclass_{classes}.png')

    # Inter-class tsne
    batch_size = int(batch_size/4)
    inter_class_img = []
    col_labels = []
    col_dict = {'flower': 'blue', 'fish' : 'green', 'gravel' : 'black', 'sugar' : 'pink'}
    for lab in ['flower', 'fish', 'sugar', 'gravel']:
        autoencoder = torch.load(os.path.join(os.getcwd(),'weights',f'{classes}_Autoencodercurrent_model.pth'), map_location = device)
        autoencoder.eval()
        with torch.no_grad():
            for _, data in enumerate(train_dataloader):
                x, _ = data
                x = x.to(device)
                _ = autoencoder.forward(x)
                latent = autoencoder.latent.detach().cpu()
                inter_class_img.append(latent)
                col_labels += [col_dict[lab] for i in range(latent.shape[0])]
                break
    inter_class_img_stack = torch.cat(inter_class_img)
    perplexities = np.arange(5, 60, 10)
    fig = plt.figure(figsize=(10,10), dpi= 100)
    fig.suptitle('Inter-class t-sne')
    for i in range(len(perplexities)):
        X_embedded = TSNE(n_components=2, perplexity = perplexities[i]).fit_transform(inter_class_img_stack)
        print(f'TSNE built for inter-class, perplexity: {perplexities[i]}')
        x = X_embedded[:,0]
        y = X_embedded[:,1]
        ax = fig.add_subplot(2, 3, i+1)
        ax.scatter(x, y, c = col_labels, alpha = 0.3)
        ax.set_title(f'Perplexity: {perplexities[i]}')
    fig.savefig(f'interclass.png')