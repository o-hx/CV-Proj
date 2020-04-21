import torch
import numpy as np
import logging
import os
import time
import segmentation_models_pytorch as smp
import torchvision

from utils.data_classifier import prep_classification_data
from utils.data import prepare_dataloader, get_augmentations
from utils.train import train_model, cluster, plot_clusters
from utils.misc import upload_google_sheets, get_module_name, log_print
from utils.tsne import get_dataloader
from models.autoencoder import Autoencoder
from models.auxillary import MSE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

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
    batch_size = 4
    img_size = (287, 287)
    start_lr = 0.001
    total_epochs = 10
    k = 5
    col_dict = {'flower': 'blue', 'fish' : 'green', 'gravel' : 'black', 'sugar' : 'pink'}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    intra = False
    inter = True

    if intra:
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
                    img, _, orig_img = data
                    img = img.to(device)
                    _ = autoencoder.forward(img)
                    latent = autoencoder.latent.detach().cpu().numpy()
                    break
                tsne_output_all_four = []
                col_labels = []
                perplexities = np.arange(5, 60, 10)
                fig = plt.figure(figsize=(10,10), dpi= 100)
                fig2,ax2 = plt.subplots(5,5, figsize = (25,25))
                fig.suptitle(classes)
                counter = 0
                for i in range(len(perplexities)):
                    X_embedded = TSNE(n_components=2, perplexity = perplexities[i]).fit_transform(latent)
                    print(f'TSNE built for {classes} intra-class, perplexity: {perplexities[i]}')
                    x = X_embedded[:,0]
                    y = X_embedded[:,1]
                    outliers = np.where( ((x > np.percentile(x, 99)) & (y > np.percentile(x, 99))) | ((x < np.percentile(x,1))  & (y < np.percentile(x,1))) )
                    for idx in range(len(list(outliers[0]))):
                        if counter >= 25:
                            break
                        ax2[counter//5,counter % 5].imshow(orig_img[list(outliers[0])[idx]].permute(1, 2, 0).cpu().detach().numpy())
                        counter += 1
                    ax = fig.add_subplot(2, 3, i+1)
                    ax.scatter(x, y, alpha = 0.3)
                    ax.set_title(f'Perplexity: {perplexities[i]}')
                fig.savefig(f'intraclass_{classes}.png')
                fig2.savefig(f'outliers_{classes}.png')

    if inter:
        # # Inter-class tsne
        batch_size = int(batch_size/4)
        inter_class_img = []
        col_labels = []
        inter_class_orig_img = []
        col_dict = {'flower': 'blue', 'fish' : 'green', 'gravel' : 'black', 'sugar' : 'pink'}
        
        classifier = torch.load(os.path.join(os.getcwd(),'weights','flower_gravel_sugar_fishclassifier_EfficientNetbest_model.pth'), map_location = device)
        classifier.eval()
        for classes in ['flower', 'fish', 'sugar', 'gravel']:
            train_dataloader, validation_dataloader = get_dataloader(df_filepath = df_filepath,
                                                        train_image_filepath = train_image_filepath,
                                                        img_size = (384, 576),
                                                        label = classes,
                                                        normalise = True,
                                                        batch_size = batch_size
                                                        )
        
         # register hook to access to features in forward pass
            features = []
            def hook(module, input, output):
                N,C,H,W = output.shape
                output = output.reshape(N, -1)
                print(output.shape)
                features.append(output.cpu().detach().numpy())
            handle = classifier._modules.get('_bn0').register_forward_hook(hook)
            with torch.no_grad():
                for _, data in enumerate(train_dataloader):
                    x, orig_img, label = data
                    x = x.to(device)
                    output = classifier(x).detach().cpu()
                    features = torch.from_numpy(features[-1])
                    inter_class_img.append(features)
                    inter_class_orig_img.append(orig_img)
                    col_labels += [col_dict[classes] for i in range(output.shape[0])]
                    break
        inter_class_img_stack = torch.cat(inter_class_img)
        perplexities = np.arange(5, 60, 10)
        fig = plt.figure(figsize=(10,10), dpi= 100)
        fig2,ax2 = plt.subplots(5,5, figsize = (25,25))
        fig.suptitle('Inter-class t-sne')
        fig2.suptitle('Outliers')
        counter = 0
        for i in range(len(perplexities)):
            X_embedded = TSNE(n_components=2, perplexity = perplexities[i]).fit_transform(inter_class_img_stack)
            print(f'TSNE built for inter-class, perplexity: {perplexities[i]}')
            x = X_embedded[:,0]
            y = X_embedded[:,1]
            outliers = np.where( ((x > np.percentile(x, 99)) | (y > np.percentile(x, 99))) | ((x < np.percentile(x,1)) | (y < np.percentile(x,1))) )
            for idx in range(len(list(outliers[0]))):
                if counter >= 25:
                    break
                index = list(outliers[0])[idx]
                ax2[counter//5,counter % 5].imshow(inter_class_orig_img[index//batch_size][index % batch_size].permute(1, 2, 0).cpu().detach().numpy())
                ax2[counter//5,counter % 5].set_title(f'Perplexity: {perplexities[i]}')
                counter += 1
            ax = fig.add_subplot(2, 3, i+1)
            ax.scatter(x, y, c = col_labels, alpha = 0.3)
            ax.set_title(f'Perplexity: {perplexities[i]}')
        fig.legend(handles=[mpatches.Patch(color=col_dict[key], label=key, alpha = 0.3) for key in col_dict.keys()])
        fig.savefig(f'graphs/interclass_tsne_classifier.png')
        fig2.savefig(f'graphs/interclass_outliers.png')