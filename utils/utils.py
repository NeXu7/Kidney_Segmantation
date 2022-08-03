from datetime import datetime
import time
import os

from skimage import io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tifffile as tf


class MetricContainer:
    def __init__(self, metrics: dict):
        self.losses = {"train": [],
                       "test": []}
        self.batch_losses = {"train": [],
                             "test": []}
        self.metrics = metrics
        self.metric_history = {key: [] for key in metrics.keys()}
        self.metric_batch_history = {key: [] for key in self.metrics.keys()}

    def batch_loss(self, key, loss):
        self.batch_losses[key].append(loss)

    def batch_metrics(self, y_pred, y_target):
        for name, metric in self.metrics.items():
            metric = metric(y_pred, y_target)
            self.metric_batch_history[name].append(metric.cpu().numpy())

    def prev_loss(self):
        return self.losses['test'][-1]

    def finish_epoch(self):
        for stage, loss in self.batch_losses.items():
            self.losses[stage].append(np.mean(loss))
        self.batch_losses = {"train": [],
                             "test": []}
        for metric_name, metric in self.metric_batch_history.items():
            self.metric_history[metric_name].append(np.mean(metric))
        self.metric_batch_history = {key: [] for key in self.metrics.keys()}

    def get_stat(self):
        return {**self.losses, **self.metric_history}


class SlideDataset(Dataset):
    def __init__(self, slide, mask, image_size, iteration_on_slide,
                 transforms=None):
        self.slide = tf.imread(slide)
        self.slide_mask = tf.imread(mask)
        self.image_size = image_size
        self.iteration_on_slide = iteration_on_slide
        self.max_y = self.slide.shape[0] - image_size
        self.max_x =self.slide.shape[1] - image_size
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = ToTensorV2()

    def __len__(self):
        return self.iteration_on_slide

    def __getitem__(self, index):
        x = np.random.randint(0, self.max_x)
        y = np.random.randint(0, self.max_y)
        image = self.slide[y:y+self.image_size, x:x+self.image_size]
        mask = self.slide_mask[y:y+self.image_size, x:x+self.image_size]
        augmented = self.transforms(image=image, mask=mask)
        return augmented["image"].float(), augmented["mask"].long()

class ClassifierTrainer:
    def __init__(self, model, loss_function, train_dataloader, test_dataloader,
                 metric_container, optimizer=None, scheduler=None,
                 optimizer_param=None, scheduler_param=None,
                 device=None, checkpoint_path=None):
        self.model = model
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if optimizer is not None:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_param)
        else:
            self.optimizer = None
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **scheduler_param)
        else:
            self.scheduler = None
        self.loss_function = loss_function
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.metric_container = metric_container
        self.epoch_trained = 0
        self.min_test_loss = np.inf
        self.best_model = None
        if checkpoint_path is None:
            self.checkpoint_path = ""
        else:
            self.checkpoint_path = checkpoint_path

    def set_optimizer(self, optimizer, optimizer_param,
                      scheduler=None, scheduler_param=None):
        self.optimizer = optimizer(self.model.parameters(), **optimizer_param)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **scheduler_param)
        else:
            self.scheduler = None

    def train_batch_step(self, batch):
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        self.optimizer.zero_grad()
        predict = self.model(X_batch)
        predict_loss = self.loss_function(predict, y_batch)
        predict_loss.backward()
        self.optimizer.step()
        self.metric_container.batch_loss("train", predict_loss.item())

    def test_batch_step(self, batch):
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        predict = self.model(X_batch)
        predict_loss = self.loss_function(predict, y_batch)
        self.metric_container.batch_loss("test", predict_loss.item())
        self.metric_container.batch_metrics(predict, y_batch)

    def set_checkpoint_path(self):
        path = datetime.now().strftime("%d.%m_%H%M%S")
        path = os.path.join("train_log", path)
        os.mkdir(path)
        self.checkpoint_path = path

    def print_score(self):
        for name, metric in self.metric_container.get_stat().items():
            print(f"{name}: {metric[-1]}")

    def fit(self, num_epoch=100, silent=False):
        self.model.to(self.device)
        print(f"Device: {self.device}")
        for epoch in range(self.epoch_trained, num_epoch):
            self.model.train()
            print("#" * 15)
            print(f"Epoch: {epoch + 1}/{num_epoch}")
            print("Train:")
            for batch in self.train_dataloader:
                self.train_batch_step(batch)

            self.model.eval()
            print("Test:")
            with torch.no_grad():
                for batch in self.test_dataloader:
                    self.test_batch_step(batch)

            self.metric_container.finish_epoch()
            if not silent:
                self.print_score()
            print("#" * 15)

            if self.min_test_loss > self.metric_container.prev_loss():
                self.min_test_loss = self.metric_container.prev_loss()
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.model.to(self.device)

            if self.scheduler is not None:
                self.scheduler.step()
            self.epoch_trained += 1
            if self.epoch_trained % 10 == 0:
                self.make_checkpoint()

    def get_best_model(self):
        return self.model.load_state_dict(self.best_model)

    def make_checkpoint(self):
        torch.save({
            "epoch": self.epoch_trained,
            "min_test_loss": self.min_test_loss,
            "model": self.best_model,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            **self.metric_container.get_stat()
        }, os.path.join(self.checkpoint_path, "checkpoint.pt"))
        torch.save(self.best_model, os.path.join(self.checkpoint_path, "best_model.pt"))

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.best_model = checkpoint["model"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.epoch_trained = checkpoint["epoch"]
            self.min_test_loss = checkpoint["min_test_loss"]
            print("Checkpoint loaded")
        except Exception as e:
            print(e)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

