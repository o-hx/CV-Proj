# Code adapted from SMP. Add Scheduler
# Implement Logging
import sys
import os
import numpy as np
import torch
import datetime as dt
import matplotlib.pyplot as plt
import time

from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, logger = None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.logger = logger

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

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

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
                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        cumulative_logs = {k: v.sum/v.n for k, v in metrics_meters.items()}
        cumulative_logs['loss'] = loss_meter.sum/loss_meter.n
        log_print(" ".join([f"{k}:{v:.4f}" for k, v in cumulative_logs.items()]), self.logger, log_only = True)

        return cumulative_logs

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, logger = None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            logger=logger
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
    def __init__(self, model, loss, metrics, device='cpu', verbose=True, logger = None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            logger = logger
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

def train_model(train_dataloader,
                validation_dataloader,
                model,
                loss,
                metrics,
                optimizer,
                scheduler = None,
                batch_size = 1,
                num_epochs = 12,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                model_save_path = os.path.join(os.getcwd(),'weights'),
                model_save_prefix = '',
                plots_save_path = os.path.join(os.getcwd(),'plots')
                ):

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
    )

    valid_epoch = ValidEpoch(
        model = model, 
        loss = loss, 
        metrics = metrics,
        device = device,
        verbose = verbose,
        logger = logger,
    )

    # Record for plotting
    metric_names = [metric.__name__ for metric in metrics]
    losses = {'train':[],'val':[]}
    metric_values = {'train':{name:[] for name in metric_names},'val':{name:[] for name in metric_names}}

    # Run Epochs
    best_perfmeasure = 0
    best_epoch = -1
    start_time = dt.datetime.now()
    log_print('Training model...', logger)

    for epoch in range(num_epochs):
        log_print(f'\nEpoch: {epoch}', logger)

        train_logs = train_epoch.run(train_dataloader)
        losses['train'].append(train_logs['loss'])
        for metric in metric_names:
            metric_values['train'][metric].append(train_logs[metric])

        valid_logs = valid_epoch.run(validation_dataloader)
        losses['val'].append(valid_logs['loss'])
        for metric in metric_names:
            metric_values['val'][metric].append(valid_logs[metric])

        if scheduler is not None:
            scheduler.step()
            log_print(f"Next Epoch Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}", logger)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if best_perfmeasure < valid_logs[metric_names[0]]: # Right now the metric to be chosen for best_perf_measure is always the first metric
            best_perfmeasure = valid_logs[metric_names[0]]
            best_epoch = epoch

            torch.save(model, os.path.join(model_save_path,model_save_prefix + 'best_model.pth'))
            log_print('Best Model Saved', logger)

        torch.save(model, os.path.join(model_save_path,model_save_prefix + 'current_model.pth'))
        log_print('Current Model Saved', logger)

    log_print(f'Best epoch: {best_epoch} Best Performance Measure: {best_perfmeasure:.5f}', logger)
    log_print(f'Time Taken to train: {dt.datetime.now()-start_time}', logger)

    # Implement plotting feature
    fig, ax = plt.subplots(1,(1+len(metrics)), figsize = (5*(1+len(metrics)),5))
    fig.suptitle(f"Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']}, Max Epochs: {num_epochs} Batch size: {batch_size}, Metric: {metrics[0].__name__}")

    ax[0].set_title('Loss Value')
    ax[0].plot(losses['train'], color = 'skyblue', label="Training Loss")
    ax[0].plot(losses['val'], color = 'orange', label = "Validation Loss")
    ax[0].legend()

    for idx, metric_name in enumerate(metric_names):
        ax[idx+1].set_title(metric_name)
        ax[idx+1].plot(metric_values['train'][metric_name], color = 'skyblue', label=f"Training {metric_name}")
        ax[idx+1].plot(metric_values['val'][metric_name], color = 'orange', label=f"Validation {metric_name}")
        ax[idx+1].legend()
    
    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)
    plt.savefig(os.path.join(plots_save_path,"nn_training_" + str(time.ctime()).replace(':','').replace('  ',' ').replace(' ','_') + ".png"))
    log_print('Plot Saved', logger)

def test_model(test_dataloader,
                model,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                logger = None,
                verbose = True,
                predictions_save_path = os.path.join(os.getcwd(),'predictions'),
                ):

    log_print('Predicting on test set...', logger)

    if os.path.exists(model_save_path):
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
    else:
        raise ImportError('Model weights do not exist')

def validate_and_plot(validation_dataloader,
                    model,
                    num_plots = 10,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    logger = None,
                    verbose = True,
                    plots_save_path = os.path.join(os.getcwd(),'prediction_plots'),
                    prefix = 'Val'
                    ):

    log_print('Validating Model...', logger)
    cloud_titles = ['Sugar','Flower','Fish','Gravel']

    if torch.cuda.is_available():
        model.cuda()

    for data_idx, data in enumerate(validation_dataloader):
        if data_idx > num_plots-1:
            break
        inputs = data[0].to(device)
        masks_pred = model(inputs)

        # Only take the first image in the batch
        img = inputs[0].detach().cpu()
        masks = data[1][0].detach().cpu()
        masks_pred = masks_pred[0].detach().cpu()
        
        img = (img*torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)+torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)).numpy().transpose((1,2,0))
        fig, ax = plt.subplots(2,4, figsize=(27, 10))

        # Plot the picture and the masks
        for idx in range(4):
            ax[0,idx].imshow(img)
            ax[0,idx].imshow(masks[idx], alpha=0.3, cmap='gray')
            ax[0,idx].set_title(cloud_titles[idx] + ' Ground Truth', fontsize=24)

        # Plot the picture and the masks
        for idx in range(4):
            ax[1,idx].imshow(img)
            ax[1,idx].imshow(masks_pred[idx], alpha=0.3, cmap='gray')
            ax[1,idx].set_title(cloud_titles[idx] + ' Predicted', fontsize=24)

        current_time = str(dt.datetime.now())[0:10].replace('-','_')
        if not os.path.exists(os.path.join(plots_save_path,current_time)):
            os.makedirs(os.path.join(plots_save_path,current_time))
        plt.savefig(os.path.join(plots_save_path,current_time, f"{prefix}_{data_idx}.png"))
        plt.close()
        log_print(f'Plot {data_idx} saved', logger)

    log_print(f'Validation completed', logger)
