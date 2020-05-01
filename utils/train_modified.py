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
from models.auxillary import Accuracy

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
        self.loss[0].to(self.device)
        self.loss[1].to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, z):
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
        metrics_meters['overall_c_accuracy'] = AverageValueMeter()
        accuracy = Accuracy(threshold=0.5)
        confusion_matrices_epoch = []
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            # Run for 1 epoch
            for x, y, z in iterator:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                loss_value, y_pred, z_pred = self.batch_update(x, y, z)
                # update loss logs
                loss_meter.add(loss_value)
                loss_logs = {self.loss[0].__name__: loss_meter.mean}
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

                accuracy_value = accuracy(z_pred, z).cpu().detach().numpy()
                metrics_meters['overall_c_accuracy'].add(accuracy_value)
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

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        prediction, class_preds = self.model.forward(x)
        loss = self.loss[0](prediction, y) + self.loss[1](class_preds, z)
        loss.backward()
        self.optimizer.step()
        loss_value = loss.cpu().item()
        assert not np.isnan(loss_value), 'Loss cannot be NaN. Please restart'
        return loss_value, prediction, class_preds

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

    def batch_update(self, x, y, z):
        with torch.no_grad():
            prediction, class_preds = self.model.forward(x)
            seg_loss = self.loss[0](prediction, y)

            clas_loss = self.loss[1](class_preds, z)
            loss_value = seg_loss.cpu().item() + clas_loss.cpu().item()
        return loss_value, prediction, class_preds

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