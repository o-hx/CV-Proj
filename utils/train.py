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

from sklearn.metrics import confusion_matrix
from copy import deepcopy
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from collections import OrderedDict

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, logger = None, classes = ['sugar','flower','fish','gravel'], enable_class_wise_metrics = True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.logger = logger
        self.classes = classes
        self.enable_class_wise_metrics = enable_class_wise_metrics

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

    def get_confusion_matrix(self, y_pred, y, threshold = 0.5):
        # Shape of y and y_pred = (bs, class, height, width)
        # Takes in y and y_pred and returns a class * [tn, fp, fn, tp]  array
        # Remember to threshold the values of y_pred first which are probabilities
        y = y.cpu().detach().numpy().astype(int)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.where(y_pred > threshold, 1, 0)

        bs, classes, height, width = y.shape
        y = np.transpose(y, [1,0,2,3]).reshape(classes, -1)        
        y_pred = np.transpose(y_pred, [1,0,2,3]).reshape(classes, -1)

        cm = []
        for clas in range(classes):
            tn, fp, fn, tp = confusion_matrix(y[clas, :], y_pred[clas, :]).ravel()
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

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            # Run for 1 epoch
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[f'{metric_fn.__name__}_overall'].add(metric_value)

                    if self.enable_class_wise_metrics:
                        for i in range(0,len(self.classes)):
                            metric_value = metric_fn(y_pred[:,i,:,:], y[:,i,:,:]).cpu().detach().numpy()
                            metrics_meters[f'{metric_fn.__name__}_{self.classes[i]}'].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items() if 'overall' in k}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                # compute confusion matrix
                confusion_matrix = self.get_confusion_matrix(y_pred, y)

        cumulative_logs = {k: v.sum/v.n for k, v in metrics_meters.items()}
        cumulative_logs['loss'] = loss_meter.sum/loss_meter.n
        log_print(" ".join([f"{k}:{v:.4f}" for k, v in cumulative_logs.items()]), self.logger)
        for i in range(len(self.classes)):
            log_print(f"Confusion Matrix of {self.classes[i]}, TN: {confusion_matrix[i,0]}. FP: {confusion_matrix[i,1]}, FN: {confusion_matrix[i,2]}, TP: {confusion_matrix[i,3]}", self.logger)
        return cumulative_logs, confusion_matrix

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, logger = None, classes = ['sugar','flower','fish','gravel'], enable_class_wise_metrics = True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            logger=logger,
            classes = classes,
            enable_class_wise_metrics = enable_class_wise_metrics
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
        return loss, prediction

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True, logger = None, classes = ['sugar','flower','fish','gravel'], enable_class_wise_metrics = True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            logger = logger,
            classes = classes,
            enable_class_wise_metrics = enable_class_wise_metrics
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

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
                classes = ['sugar','flower','fish','gravel'],
                logger = None,
                verbose = True,
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
        classes = classes
    )

    valid_epoch = ValidEpoch(
        model = model, 
        loss = loss, 
        metrics = metrics,
        device = device,
        verbose = verbose,
        logger = logger,
        classes = classes
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
    # Print the confusion matrix final score
    # You need to sum up the confusion matrices along one axis first
    # Print for train, val and val_no_empty
    #log_print(f"TP: {confusion_matrix[0,0]}. FP: {confusion_matrix[0,1]}, FN: {confusion_matrix[1,0]}, TN: {confusion_matrix[1,1]}", self.logger)

    # Once done save the summed confusion matrices back into the confusion_matrices dictionary

    # Implement plotting feature
    # The code is only meant to work with the top 2 validation datasets, and no more
    fig, ax = plt.subplots(1,(1+len(metrics)), figsize = (5*(1+len(metrics)),5))
    fig.suptitle(f"Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}, Max Epochs: {num_epochs} Batch size: {batch_size}, Metric: {metrics[0].__name__}")

    ax[0].set_title('Loss Value')
    ax[0].plot(losses['train'], color = 'skyblue', label="Training Loss")
    ax[0].plot(losses['val'][0], color = 'orange', label = "Validation Loss")
    ax[0].plot(losses['val'][1], color = 'green', label = "Validation Loss 2")
    ax[0].legend()

    idx = 0
    for _, metric_name in enumerate(metric_names):
        if 'overall' in metric_name: # Only plot for the overall metric and not all metrics
            ax[idx+1].set_title(metric_name)
            ax[idx+1].plot(metric_values['train'][metric_name], color = 'skyblue', label=f"Training {metric_name}")
            ax[idx+1].plot(metric_values['val'][0][metric_name], color = 'orange', label=f"Validation 1 {metric_name}")
            ax[idx+1].plot(metric_values['val'][1][metric_name], color = 'green', label=f"Validation 2 {metric_name}")
            ax[idx+1].legend()
            idx += 1
    
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)
    plt.savefig(os.path.join(plots_save_path,"nn_training_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    log_print('Plot Saved', logger)

    return losses, metric_values, best_epoch#, confusion_matrices

def test_model(test_dataloader,
                model,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                predictions_save_path = os.path.join(os.getcwd(),'predictions'),
                ):

    log_print('Predicting on test set...', logger)

    if torch.cuda.is_available():
        model.cuda()

    all_outputs = torch.tensor([], device=device)

    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            inputs = data[0].to(device)
            outputs = model(inputs)
            all_outputs = torch.cat((all_outputs, outputs), 0)
    
    log_print('Saving predictions...', logger)
    np.save(os.path.join(predictions_save_path,str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_')), all_outputs)

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