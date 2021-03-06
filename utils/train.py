# Code adapted from SMP. Add Scheduler
# Implement Logging
import sys
import os
import numpy as np
import torch
import datetime as dt
import matplotlib.pyplot as plt
import time
import torchvision
import random
import seaborn as sns
import pickle

from segmentation_models_pytorch.utils.metrics import IoU
from sklearn.metrics import roc_curve, roc_auc_score, silhouette_score
from copy import deepcopy
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from collections import OrderedDict
from utils.misc import log_print, compute_cm_binary, get_iou_score
from sklearn.cluster import KMeans

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, logger = None, classes = ['sugar','flower','fish','gravel'], enable_class_wise_metrics = True, autoencoder = False):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.logger = logger
        self.classes = classes
        self.enable_class_wise_metrics = enable_class_wise_metrics
        self.autoencoder = autoencoder

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    @staticmethod
    def get_confusion_matrix(y_pred, y, threshold = 0.5):

        # Shape of y and y_pred = (bs, class, height, width)
        # Takes in y and y_pred and returns a class * [tn, fp, fn, tp]  array
        # Remember to threshold the values of y_pred first which are probabilities
        y = y.cpu().detach().numpy().astype(int)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.where(y_pred > threshold, 1, 0)

        if len(y_pred.shape) == 4:
            bs, classes, height, width = y.shape
            y = np.transpose(y, [1,0,2,3]).reshape(classes, -1)        
            y_pred = np.transpose(y_pred, [1,0,2,3]).reshape(classes, -1)
        else:
            _, classes = y.shape
            y = y.transpose()
            y_pred = y_pred.transpose()

        cm = []
        for clas in range(classes):
            tn, fp, fn, tp = compute_cm_binary(y[clas,:], y_pred[clas,:])
            cm.append([tn, fp, fn, tp])
        return np.array(cm)

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        if self.enable_class_wise_metrics:
            metric_meter_classes = self.classes + ['overall']
        else:
            metric_meter_classes = ['overall']

        metrics_meters = {f'{metric.__name__}_{_class}': AverageValueMeter() for metric in self.metrics for _class in metric_meter_classes}
        confusion_matrices_epoch = []
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            # Run for 1 epoch
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss_value, y_pred = self.batch_update(x, y)
                # update loss logs
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[f'{metric_fn.__name__}_overall'].add(metric_value)

                    if self.enable_class_wise_metrics:
                        for i in range(0,len(self.classes)):
                            if len(y_pred.shape) == 4:
                                metric_value = metric_fn(y_pred[:,i,:,:], y[:,i,:,:]).cpu().detach().numpy()
                            elif len(y_pred.shape) == 2:
                                metric_value = metric_fn(y_pred[:,i], y[:,i]).cpu().detach().numpy()
                            else:
                                raise NotImplementedError('Shape of y_pred must have length 2 or 4')
                            metrics_meters[f'{metric_fn.__name__}_{self.classes[i]}'].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items() if 'overall' in k}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                # compute confusion matrix
                confusion_matrices_epoch.append(self.get_confusion_matrix(y_pred, y))

        confusion_matrices_epoch = np.array(confusion_matrices_epoch).sum(axis = 0)
        cumulative_logs = {k: v.sum/v.n for k, v in metrics_meters.items()}
        cumulative_logs['loss'] = loss_meter.sum/loss_meter.n
        log_print(" ".join([f"{k}:{v:.4f}" for k, v in cumulative_logs.items()]), self.logger)
        if not self.autoencoder:
            for i in range(len(self.classes)):
                log_print(f"Confusion Matrix of {self.classes[i]}, TN: {confusion_matrices_epoch[i,0]}. FP: {confusion_matrices_epoch[i,1]}, FN: {confusion_matrices_epoch[i,2]}, TP: {confusion_matrices_epoch[i,3]}", self.logger)
        return cumulative_logs, confusion_matrices_epoch

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, logger = None, classes = ['sugar','flower','fish','gravel'], enable_class_wise_metrics = True, autoencoder = False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            logger=logger,
            classes = classes,
            enable_class_wise_metrics = enable_class_wise_metrics,
            autoencoder = autoencoder
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        loss_value = loss.cpu().detach().numpy()
        assert not np.isnan(loss_value), 'Loss cannot be NaN. Please restart'
        return loss_value, prediction

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True, logger = None, classes = ['sugar','flower','fish','gravel'], enable_class_wise_metrics = True, autoencoder = False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            logger = logger,
            classes = classes,
            enable_class_wise_metrics = enable_class_wise_metrics,
            autoencoder = autoencoder
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
            loss_value = loss.cpu().detach().numpy()
        return loss_value, prediction

def plot_loss_metrics(validation_dataloader_list, losses, metrics, metric_values, metric_names, lr, num_epochs, batch_size, plots_save_path, start_time, logger = None):
    # Implement plotting feature
    # The code is only meant to work with the top 2 validation datasets, and no more
    fig, ax = plt.subplots(1,(1+len(metrics)), figsize = (5*(1+len(metrics)),5))
    fig.suptitle(f"Learning Rate: {lr:.5f}, Max Epochs: {num_epochs} Batch size: {batch_size}, Metric: {metrics[0].__name__}")

    ax[0].set_title('Loss Value')
    ax[0].plot(losses['train'], color = 'skyblue', label="Training Loss")
    ax[0].plot(losses['val'][0], color = 'orange', label = "Validation Loss")
    if len(validation_dataloader_list) > 1:
        ax[0].plot(losses['val'][1], color = 'green', label = "Validation Loss 2")
    ax[0].legend()

    idx = 0
    for _, metric_name in enumerate(metric_names):
        if 'overall' in metric_name: # Only plot for the overall metric and not all metrics
            ax[idx+1].set_title(metric_name)
            ax[idx+1].plot(metric_values['train'][metric_name], color = 'skyblue', label=f"Training {metric_name}")
            ax[idx+1].plot(metric_values['val'][0][metric_name], color = 'orange', label=f"Validation 1 {metric_name}")
            if len(validation_dataloader_list) > 1:
                ax[idx+1].plot(metric_values['val'][1][metric_name], color = 'green', label=f"Validation 2 {metric_name}")
            ax[idx+1].legend()
            idx += 1
    
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)
    plt.savefig(os.path.join(plots_save_path,"nn_training_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    log_print('Metric & Loss Plot Saved', logger)
    plt.close()

def plot_cm(confusion_matrices, classes, validation_dataloader_list, start_time, plots_save_path, colors = ['black','orange','red','green'], logger = None):
    # Plot another plot for the confusion matrices
    fig, ax = plt.subplots(len(classes),len(validation_dataloader_list)+1, figsize=(10*(len(validation_dataloader_list)+1), 7*len(classes)))
    for class_idx, _class in enumerate(classes):
        for clx_idx, classification in enumerate(['TN','FP','FN','TP']):
            if len(classes) > 1:
                ax[class_idx,0].plot([cm[class_idx,clx_idx] for cm in confusion_matrices['train']], color = colors[clx_idx], label=f"{classification}")
                ax[class_idx,0].set_title(f'Training Confusion Matrix {_class}', fontsize=12)
                ax[class_idx,0].legend()

                ax[class_idx,1].plot([cm[class_idx,clx_idx] for cm in confusion_matrices['val'][0]], color = colors[clx_idx], label=f"{classification}")
                ax[class_idx,1].set_title(f'Val 1 Confusion Matrix {_class}', fontsize=12)
                ax[class_idx,1].legend()
                if len(validation_dataloader_list) > 1:
                    ax[class_idx,2].plot([cm[class_idx,clx_idx] for cm in confusion_matrices['val'][1]], color = colors[clx_idx], label=f"{classification}")
                    ax[class_idx,2].set_title(f'Val 2 Confusion Matrix {_class}', fontsize=12)
                    ax[class_idx,2].legend()
            else:
                ax[0].plot([cm[class_idx,clx_idx] for cm in confusion_matrices['train']], color = colors[clx_idx], label=f"{classification}")
                ax[0].set_title(f'Training Confusion Matrix {_class}', fontsize=12)
                ax[0].legend()

                ax[1].plot([cm[class_idx,clx_idx] for cm in confusion_matrices['val'][0]], color = colors[clx_idx], label=f"{classification}")
                ax[1].set_title(f'Val 1 Confusion Matrix {_class}', fontsize=12)
                ax[1].legend()

                if len(validation_dataloader_list) > 1:
                    ax[2].plot([cm[class_idx,clx_idx] for cm in confusion_matrices['val'][1]], color = colors[clx_idx], label=f"{classification}")
                    ax[2].set_title(f'Val 2 Confusion Matrix {_class}', fontsize=12)
                    ax[2].legend()
            
    fig.suptitle(f'Confusion Matrix Plot Across Epochs', fontsize=20)
    plt.savefig(os.path.join(plots_save_path,"nn_training_cm_" + str(start_time).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    plt.close()

def plot_roc_iou(dataloader_list,
                dataloader_name_list,
                model,
                classes = ['sugar','flower','fish','gravel'],
                batch_samples = 10,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                plots_save_path = os.path.join(os.getcwd(),'roc_iou_plots')):

    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)

    log_print('Running Inference...', logger)
    picture_iou_scores = OrderedDict()

    if torch.cuda.is_available():
        model.cuda()

    y_pred = []
    y = []

    for dataloader_idx, dataloader in enumerate(dataloader_list):
        y_pred_dl = []
        y_dl = []
        with tqdm(enumerate(dataloader), desc='Inference', file=sys.stdout, disable=False, total=batch_samples) as iterator:
            for data_idx, data in iterator:
                if data_idx == batch_samples:
                    break
                inputs = data[0].to(device)
                masks_pred = model(inputs)
                masks = data[1].to(device)
                if masks_pred.shape[1] != len(classes):
                    raise Exception('Your model predicts more classes than the number of classes specified')
                y_pred_dl.append(masks_pred.cpu().detach())
                y_dl.append(masks.cpu().detach())
        y_pred_dl, y_dl = torch.cat(y_pred_dl).cpu().detach(), torch.cat(y_dl).type(torch.long).cpu().detach()
        y_pred.append(y_pred_dl)
        y.append(y_dl)

    iou_scores = []
    threshold_intervals = np.round(np.linspace(0,1,21),2).tolist()
    for dataloader_idx in range(len(dataloader_list)):
        iou_scores_dl = []
        for clas in range(len(classes)):
            temp_ls = []
            for threshold in threshold_intervals:
                iou = IoU(threshold=threshold)
                value = iou(y_pred[dataloader_idx][:,clas,:,:], y[dataloader_idx][:,clas,:,:]).item()
                # print(clas, threshold, value)
                temp_ls.append(value)
            iou_scores_dl.append(temp_ls)
        iou_scores.append(iou_scores_dl)

    # iou_scores is a list of four sublists, one for each class
    # each sublist contains distribution of iou scores across threshold intervals
    # threshold interval is from 0 to 1 inclusive, at 0.5 interval
    
    def create_roc_plot(y_pred, y, ax, classes, dl_name, colors = ['blue','green','red','orange']):
        for class_idx, clas in enumerate(classes):
            label = y[:,class_idx,:,:].flatten()
            pred = y_pred[:,class_idx,:,:].flatten()
            fpr, tpr, _ = roc_curve(label, pred)
            auc = roc_auc_score(label, pred)
            ax.set_xticks(np.arange(0, 1.05, 0.1))
            ax.plot(fpr,tpr,label=f'{clas}, auc = {auc:.3f}', color = colors[class_idx])
            ax.plot(np.linspace(0,1,11),np.linspace(0,1,11), '--', color = 'black')
        ax.legend()
        ax.set_title(f'ROC Plot for {dl_name}', fontsize=12)
    
    def create_threshold_plot(iou_scores, threshold_intervals, ax, classes, dl_name, colors = ['blue','green','red','orange']):
        for class_idx, clas in enumerate(classes):
            class_iou_scores = iou_scores[class_idx]
            best_iou_index = class_iou_scores.index(max(class_iou_scores))
            optimal_threshold = threshold_intervals[best_iou_index]
            ax.set_xticks(np.arange(0, 1.05, 0.1))
            ax.plot(threshold_intervals, class_iou_scores, label=f'{clas}, best_thresh = {optimal_threshold:.1f}')
        ax.legend()
        ax.set_title(f'IOU Plot for {dl_name}', fontsize=12)

    fig, ax = plt.subplots(2,len(dataloader_list), figsize=(7*len(dataloader_list), 10))
    for dl_idx, dl in enumerate(dataloader_list):
        if len(dataloader_list) == 1:
            create_roc_plot(y_pred[dl_idx].numpy(), y[dl_idx].numpy(), ax[0], classes, dataloader_name_list[dl_idx])
            create_threshold_plot(iou_scores[dl_idx], threshold_intervals, ax[1], classes, dataloader_name_list[dl_idx])
        else:
            create_roc_plot(y_pred[dl_idx].numpy(), y[dl_idx].numpy(), ax[0,dl_idx], classes, dataloader_name_list[dl_idx])
            create_threshold_plot(iou_scores[dl_idx], threshold_intervals, ax[1,dl_idx], classes, dataloader_name_list[dl_idx])

    current_time = str(dt.datetime.now())[0:19].replace('-','_').replace(' ','_').replace(':','_')
    plt.savefig(os.path.join(plots_save_path,"roc_iou_" + current_time + ".png"))
    log_print('ROC and IoU Plot Saved', logger)
    plt.close()

def train_model(train_dataloader,
                validation_dataloader_list,
                model,
                loss,
                metrics,
                optimizer,
                scheduler = None,
                batch_size = 1,
                num_epochs = 12,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                autoencoder = False,
                classes = ['sugar','flower','fish','gravel'],
                logger = None,
                verbose = True,
                only_validation = False,
                model_save_path = os.path.join(os.getcwd(),'weights'),
                model_save_prefix = '',
                plots_save_path = os.path.join(os.getcwd(),'plots')
                ):
    if type(validation_dataloader_list) != list:
        raise TypeError('validation_dataloader_list must be a list of validation dataloaders')

    if torch.cuda.is_available():
        log_print('Using GPU', logger)
    else:
        log_print('Using CPU', logger)
    
    # Define Epochs
    train_epoch = TrainEpoch(
        model = model,
        loss = loss, 
        metrics = metrics,
        optimizer = optimizer,
        device = device,
        verbose = verbose,
        logger = logger,
        classes = classes,
        autoencoder = autoencoder
    )

    valid_epoch = ValidEpoch(
        model = model, 
        loss = loss, 
        metrics = metrics,
        device = device,
        verbose = verbose,
        logger = logger,
        classes = classes,
        autoencoder = autoencoder
    )

    # Record for plotting
    metric_names = [f'{metric.__name__}_{_class}' for metric in metrics for _class in ['overall'] + classes]
    losses = {'train':[],'val':{idx:[] for idx in range(len(validation_dataloader_list))}}
    metric_values = {'train':{name:[] for name in metric_names},'val':{idx:{name:[] for name in metric_names} for idx in range(len(validation_dataloader_list))}}
    confusion_matrices = {'train':[],'val':{idx:[] for idx in range(len(validation_dataloader_list))}}

    # Run Epochs
    best_perfmeasure = 0
    best_epoch = -1
    start_time = dt.datetime.now()
    log_print('Training model...', logger)

    for epoch in range(num_epochs):
        log_print(f'\nEpoch: {epoch}', logger)

        if not only_validation:
            train_logs, train_cm = train_epoch.run(train_dataloader)
            losses['train'].append(train_logs['loss'])
            confusion_matrices['train'].append(train_cm)
            for metric in metric_names:
                metric_values['train'][metric].append(train_logs[metric])

        valid_logs = {}
        for valid_idx, validation_dataloader in enumerate(validation_dataloader_list):
            valid_logs[valid_idx], val_cm = valid_epoch.run(validation_dataloader)
            losses['val'][valid_idx].append(valid_logs[valid_idx]['loss'])
            confusion_matrices['val'][valid_idx].append(val_cm)
            for metric in metric_names:
                metric_values['val'][valid_idx][metric].append(valid_logs[valid_idx][metric])

        if scheduler is not None:
            scheduler.step()
            log_print(f"Next Epoch Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}", logger)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if best_perfmeasure < valid_logs[0][metric_names[0]]: # Right now the metric to be chosen for best_perf_measure is always the first metric for the first validation dataset
            best_perfmeasure = valid_logs[0][metric_names[0]]
            best_epoch = epoch

            torch.save(model, os.path.join(model_save_path,model_save_prefix + 'best_model.pth'))
            log_print('Best Model Saved', logger)

        torch.save(model, os.path.join(model_save_path,model_save_prefix + 'current_model.pth'))
        log_print('Current Model Saved', logger)

    log_print(f'Best epoch: {best_epoch} Best Performance Measure: {best_perfmeasure:.5f}', logger)
    log_print(f'Time Taken to train: {dt.datetime.now()-start_time}', logger)
    
    plot_loss_metrics(validation_dataloader_list, losses, metrics, metric_values, metric_names, optimizer.state_dict()['param_groups'][0]['lr'], num_epochs, batch_size, plots_save_path, start_time, logger = logger)
    if not autoencoder:
        plot_cm(confusion_matrices, classes, validation_dataloader_list, start_time, plots_save_path, logger = logger)
        # Sum up confusion matrix along all batches
        confusion_matrices['train'] = confusion_matrices['train'][-1]
        for valid_idx in range(len(validation_dataloader_list)):
            confusion_matrices['val'][valid_idx] = confusion_matrices['val'][valid_idx][-1]
        for i in range(len(classes)):
            log_print(f"Confusion Matrix of {classes[i]}, TN: {confusion_matrices['val'][0][i,0]}. FP: {confusion_matrices['val'][0][i,1]}, FN: {confusion_matrices['val'][0][i,2]}, TP: {confusion_matrices['val'][0][i,3]}", logger)

        return losses, metric_values, best_epoch, confusion_matrices
    else:
        return losses, metric_values, best_epoch, model

def cluster(model,
            dataloader,
            k,
            criterion,
            optimizer,
            scheduler = None,
            total_epochs = 10,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            logger = None,
            tolerance = 0.001,
            model_save_prefix = '',
            model_save_path = os.path.join(os.getcwd(),'cluster_weights')
            ):

    def target_distribution(q):
        weight = torch.pow(q,2)/q.sum(axis = 0)
        return (weight.T/weight.sum(1)).T
    
    assert k > 1, 'K Must be larger than 1'

    model.to(device)
    model.switch_to_kmeans(k)

    # Set initial KMeans cluster centres
    log_print('Computing KMeans', logger)
    kmeans = KMeans(n_clusters=k, n_init=10)
    _, data = next(enumerate(dataloader))
    y_pred_last = kmeans.fit_predict(model.encoder.forward(data[0].to(device)).detach().cpu().numpy())
    log_print('KMeans Computed', logger)
    model.k_means_weights.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
    log_print('Initial Cluster Centres Added', logger)

    for ite in range(total_epochs):
        loss_meter = AverageValueMeter()
        metric_meter = AverageValueMeter()
        y_preds_last_epoch = []
        y_preds_epoch = []
        with tqdm(dataloader, desc=f'Clustering Epoch {ite+1}', file=sys.stdout, disable=False) as iterator:
            for _, data in enumerate(iterator):
                x, _ = data
                x = x.to(device)
                optimizer.zero_grad()
                q, features = model(x)
                p = target_distribution(q).detach()

                loss = criterion(torch.log(q),p)
                loss.backward()
                optimizer.step()

                labels = torch.argmax(q, axis = 1).detach().cpu().numpy()
                y_preds_epoch.append(labels)

                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)

                metric_meter.add(silhouette_score(features.detach().cpu().numpy(), labels))

                iterator.set_postfix_str(f'DKL Loss: {loss_meter.mean:.3f} Silhouette: {metric_meter.mean:.3f}')

        y_preds_epoch = np.concatenate(y_preds_epoch)
        if ite > 0:
            delta_preds = np.sum(y_preds_epoch != y_preds_last_epoch).astype(np.float32)
            total_preds = y_preds_epoch.shape[0]
            log_print(f'Total Different Predictions: {int(delta_preds)}', logger)
            log_print(f'Total Predictions: {total_preds}', logger)
            if ite > 0 and delta_preds <= max(int(tolerance*total_preds),1):
                print('Reached tolerance threshold. Stopping training.')
                break
        y_preds_last_epoch = np.copy(y_preds_epoch)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(model, os.path.join(model_save_path,model_save_prefix + '_clustering_model.pth'))
    log_print('Current Model Saved', logger)
    
    return model

def plot_clusters(model,
                k,
                clas,
                dataloader,
                batch_size,
                original_dataloader,
                plots_save_path,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None
                ):

    images = {i:[] for i in range(k)}
    trans = torchvision.transforms.ToPILImage()
    
    with tqdm(zip(dataloader, original_dataloader), desc=f'Clustering Images', file=sys.stdout, disable=False) as iterator:
        for data, org_data in iterator:
            img_batch = data[0].to(device)
            org_img_batch = org_data[1].cpu()

            q, _ = model(img_batch)
            q = torch.argmax(q, axis = 1).detach()
            for idx, pred in enumerate(q):
                img = np.array(trans(org_img_batch[idx]))
                images[pred.item()].append(img)

    for cluster in range(k):
        min_image_number = min(36,len(images[cluster]))
        selected_images = random.sample(images[cluster],k=min_image_number)
        fig, ax = plt.subplots(6,6, figsize=(20, 20))
        l = 0
        for i in range(6):
            for j in range(6):
                if l >= min_image_number:
                    continue
                ax[i,j].imshow(selected_images[l])
                l += 1

        current_time = str(dt.datetime.now())[0:10].replace('-','_')
        if not os.path.exists(os.path.join(plots_save_path,current_time,clas)):
            os.makedirs(os.path.join(plots_save_path,current_time,clas))
        fig.suptitle(f'Cluster {cluster}', fontsize=20)
        plt.savefig(os.path.join(plots_save_path,current_time,clas, f"{cluster}.png"))
        plt.close()
        log_print(f'Plot for cluster {cluster} saved', logger)

    pickle.dump(images, open(os.path.join(plots_save_path,current_time,clas,"cluster_predictions.pk"), "wb" ) )
    log_print(f'Pickle Saved', logger)
    
    sns.barplot(x = np.arange(k), y = np.array([len(image_list) for c, image_list in images.items()]))
    plt.title('Cluster Size')
    plt.xlabel('Cluster')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(plots_save_path,current_time,clas, f"Cluster Size.png"))
    plt.close()
    log_print(f'Cluster size plot saved', logger)

def test_model(test_dataloader,
                model,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                predictions_save_path = os.path.join(os.getcwd(),'predictions'),
                ):

    log_print('Predicting on test set...', logger)
    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    if torch.cuda.is_available():
        model.cuda()

    all_outputs = []
    filepath = []

    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            inputs = data[0].to(device)
            filepath += [os.path.basename(i) for i in data[1]]
            outputs = model(inputs)
            all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs)
    
    log_print('Saving predictions...', logger)
    np.save(os.path.join(predictions_save_path,str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')), all_outputs.cpu().numpy() )
    return all_outputs.cpu().numpy(), filepath

def validate_and_plot(validation_dataloader,
                    validation_dataloader_org,
                    model,
                    metrics,
                    classes = ['sugar','flower','fish','gravel'],
                    top_n = 20,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    logger = None,
                    verbose = True,
                    plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
                    prefix = 'Val',
                    threshold = 0.5
                    ):
    # Plots the top N images in terms of IoU scores

    # Helper function to return the image, the predicted mask and the actual mask for any datapoint
    def _get_image_mask(data, model, class_idx):
        aug_data, org_data = data
        inputs = aug_data[0].to(device)
        masks_pred = model(inputs)

        img = org_data[0][0].detach().cpu()
        mask = aug_data[1][0][class_idx].cpu()
        mask_pred = masks_pred[0][class_idx].cpu()
        mask_pred = np.where(mask_pred > threshold,1,0)
        trans = torchvision.transforms.ToPILImage()
        
        img = np.array(trans(img))
        return img, mask_pred, mask

    # Helper function to plot mask and original pictures
    def _plot_topn(rank, metric_info, low_high_str):
        fig, ax = plt.subplots(2*len(metrics),len(classes), figsize=(7*len(classes), 10*len(metrics)))
        for class_idx, _class in enumerate(classes):
            for m_idx, metric in enumerate(metrics):
                # Plot the Ground Truth
                if len(classes) > 1:
                    ax[m_idx*2,class_idx].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2,class_idx].imshow(metric_info[_class,metric.__name__][rank][2], alpha=0.3, cmap='gray')
                    ax[m_idx*2,class_idx].set_title(f'{_class} Ground Truth {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)

                    # Plot the picture and the masks
                    ax[m_idx*2+1,class_idx].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2+1,class_idx].imshow(metric_info[_class,metric.__name__][rank][1], alpha=0.3, cmap='gray')
                    ax[m_idx*2+1,class_idx].set_title(f'{_class} Prediction {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)
                else:
                    ax[m_idx*2].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2].imshow(metric_info[_class,metric.__name__][rank][2], alpha=0.3, cmap='gray')
                    ax[m_idx*2].set_title(f'{_class} Ground Truth {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)

                    # Plot the picture and the masks
                    ax[m_idx*2+1].imshow(metric_info[_class,metric.__name__][rank][0])
                    ax[m_idx*2+1].imshow(metric_info[_class,metric.__name__][rank][1], alpha=0.3, cmap='gray')
                    ax[m_idx*2+1].set_title(f'{_class} Prediction {metric.__name__}:{metric_info[_class,metric.__name__][rank][3]:.3f}', fontsize=12)
                
        current_time = str(dt.datetime.now())[0:10].replace('-','_')
        if not os.path.exists(os.path.join(plots_save_path,current_time)):
            os.makedirs(os.path.join(plots_save_path,current_time))
        fig.suptitle(f'{low_high_str} {rank+1}', fontsize=20)
        plt.savefig(os.path.join(plots_save_path,current_time, f"{prefix}_{low_high_str}_{rank+1}.png"))
        plt.close()
        log_print(f'Plot {low_high_str} {rank+1} saved', logger)

    log_print('Running Inference...', logger)
    picture_iou_scores = OrderedDict()

    if torch.cuda.is_available():
        model.cuda()

    with tqdm(validation_dataloader, desc='Inference Round 1', file=sys.stdout, disable=False) as iterator:
        for data_idx, data in enumerate(iterator):
            inputs = data[0].to(device)
            masks_pred = model(inputs)

            # Make sure batch size is 1
            assert len(inputs) == 1
            masks = data[1][0].to(device)
            masks_pred = masks_pred[0]
            if masks_pred.shape[0] != len(classes):
                raise Exception('Your model predicts more classes than the number of classes specified')

            for class_idx in range(len(classes)):
                for metric in metrics:
                    picture_iou_scores[data_idx,classes[class_idx],metric.__name__] = float(metric(masks_pred[class_idx,:,:], masks[class_idx,:,:]).cpu().detach().numpy())
    
    # Only do it for the first metric
    lowest = {}
    highest = {}
    for _class in classes:
        for metric in metrics:
            sorted_iou_scores = sorted(filter(lambda x:x[0][1] == _class and x[0][2] == metric.__name__, picture_iou_scores.items()), key = lambda x:x[1])
            sorted_iou_scores = [(idx, value[0][0], value[1]) for idx, value in enumerate(sorted_iou_scores)]
            lowest[_class,metric.__name__] = sorted_iou_scores[:top_n]
            lowest[_class,metric.__name__] = {value[1]:(value[0],value[2]) for value in lowest[_class,metric.__name__]}
            highest[_class,metric.__name__] = sorted_iou_scores[-1*top_n:]
            highest[_class,metric.__name__] = {value[1]:(idx,value[2]) for idx, value in enumerate(highest[_class,metric.__name__])}

    lowest_processed = {(_class, metric.__name__):[0 for _ in range(top_n)] for _class in classes for metric in metrics}
    highest_processed = {(_class, metric.__name__):[0 for _ in range(top_n)] for _class in classes for metric in metrics}

    with tqdm(zip(validation_dataloader,validation_dataloader_org), desc='Inference Round 2', file=sys.stdout, disable=False) as iterator:
        for data_idx, data in enumerate(iterator):
            for class_idx, _class in enumerate(classes):
                for metric in metrics:
                    if data_idx in lowest[_class,metric.__name__].keys():
                        img, mask_pred, mask = _get_image_mask(data, model, class_idx)
                        rank = lowest[_class,metric.__name__][data_idx][0]
                        metric_value = lowest[_class,metric.__name__][data_idx][1]
                        lowest_processed[_class,metric.__name__][rank] = (img, mask_pred, mask, metric_value)
                    
                    elif data_idx in highest[_class,metric.__name__].keys():
                        img, mask_pred, mask = _get_image_mask(data, model, class_idx)
                        rank = highest[_class,metric.__name__][data_idx][0]
                        metric_value = highest[_class,metric.__name__][data_idx][1]
                        highest_processed[_class,metric.__name__][rank] = (img, mask_pred, mask, metric_value)

    for rank in range(top_n):
        _plot_topn(rank, lowest_processed, 'Lowest')
        _plot_topn(rank, highest_processed, 'Highest')

    log_print(f'Validation completed', logger)