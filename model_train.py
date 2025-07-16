import copy
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import logging

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_utils import *
from model_diff import VDM_Tools, DDPM_Tools
from model_test import calculate_metrics
from itertools import cycle


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            optimizer: optim,
            result_path: str,
            train_path: str,
            model_path: str,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            diffusion: DDPM_Tools,
            ema: bool = True,
            **kwargs
    ):
        super().__init__()

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.device = device
        self.mse = nn.MSELoss()
        self.ema = ema
        self.epochs = kwargs.get("epochs")
        self.result_path = result_path
        self.train_path = train_path
        self.model_path = model_path
        self.threshold_training = kwargs.get("threshold_training")
        self.threshold = kwargs.get("threshold")
        self.sample_number = kwargs.get("sample_number")
        self.sample_epoch = kwargs.get("sample_epoch")
        self.ema_decay = kwargs.get("ema_decay")
        self.clip_grad = kwargs.get("clip_grad")
        self.resolution = kwargs.get("resolution")
        self.noise_schedule = kwargs.get("noise_schedule")
        self.loss = kwargs.get("loss")
        self.physics_informed = kwargs.get("physics_informed")

        self.train_losses = []
        self.val_losses = []
        self.ssim_values = []
        self.mae_values = []

        self.best_val_loss = float('inf')
        self.best_model_checkpoint = None
        self.best_model_epoch = None

        self.ema_model = copy.deepcopy(model)
        self.ema_model.to(self.device)
        for param in self.ema_model.parameters():
            param.detach()

        dataset_range = pd.read_csv(os.path.join(self.result_path, "dataset_range.csv"))
        self.surface_max = dataset_range["surface_max"].values[0]
        self.surface_min = dataset_range["surface_min"].values[0]
        self.side_max = dataset_range["side_max"].values[0]
        self.side_min = dataset_range["side_min"].values[0]
        self.condition_max = dataset_range["condition_max"].values[0]
        self.condition_min = dataset_range["condition_min"].values[0]
        self.gap_max = dataset_range["gap_max"].values[0]
        self.gap_min = dataset_range["gap_min"].values[0]

    def train_epoch(self):
        loss_total = 0
        self.model.train()
        pbar = tqdm(self.train_dataloader, desc="training loop", leave=False)

        for i, (targets, conditions) in enumerate(pbar):
            targets, conditions = targets.to(self.device), conditions.to(self.device)
            t = self.diffusion.sample_timesteps(targets.shape[0]).to(self.device) # l2: [0, 1000(noise step)], vlb: [0, 1)
            losses = self.diffusion.training_losses(model=self.model, x_start=targets, c=conditions, t=t)
            loss = losses["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            if self.ema == True:
                self.update_ema()

            loss_total += loss.item()
            pbar.set_postfix(loss=loss.item())

        average_loss = loss_total / len(self.train_dataloader)
        self.train_losses.append(average_loss)

    def validation_epoch(self):
        loss_total = 0
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="validation loop", leave=False)

            for i, (targets, conditions) in enumerate(pbar):
                targets, conditions = targets.to(self.device), conditions.to(self.device)
                t = self.diffusion.sample_timesteps(targets.shape[0]).to(self.device)
                losses = self.diffusion.training_losses(model=self.model, x_start=targets, c=conditions, t=t)
                loss = losses["loss"].mean()

                pbar.set_postfix(loss=loss.item())
                loss_total += loss.item()

            average_loss = loss_total / len(self.val_dataloader)
            self.val_losses.append(average_loss)
            return average_loss

    def generate_samples(self, epoch):
        targets, conditions = next(cycle(self.test_dataloader))
        targets, conditions = targets.to(self.device), conditions.to(self.device)
        targets, conditions = concat_to_batchsize(targets, self.sample_number), concat_to_batchsize(conditions, self.sample_number)

        if self.ema == True:
            generations, conditions = self.diffusion.p_sample_loop(self.ema_model, n=self.sample_number, c=conditions, resolution=self.resolution)
        else:
            generations, conditions = self.diffusion.p_sample_loop(self.model, n=self.sample_number, c=conditions, resolution=self.resolution)

        if self.physics_informed == True:
            g = ((generations.clamp(-1, 1) + 1) / 2) * (self.gap_max - self.gap_min) + self.gap_min
            t = ((targets.clamp(-1, 1) + 1) / 2) * (self.gap_max - self.gap_min) + self.gap_min
            c = ((conditions.clamp(-1, 1) + 1) / 2) * (self.condition_max - self.condition_min) + self.condition_min

            g = g + c
            t = t + c

            generations = g.clamp(self.surface_min, self.surface_max)
            generations = (generations - self.surface_min) / (self.surface_max - self.surface_min)
            generations = ((generations * 2) - 1)
            targets = t.clamp(self.surface_min, self.surface_max)
            targets = (targets - self.surface_min) / (self.surface_max - self.surface_min)
            targets = ((targets * 2) - 1)

        targets_value, targets_pixel, targets_max, targets_min = tensor_to_PIL_range(targets, self.surface_max, self.surface_min)
        generations_value, generations_pixel, generations_max, generations_min = tensor_to_PIL_range(generations, self.surface_max, self.surface_min)
        conditions_value, conditions_pixel, conditions_max, conditions_min = tensor_to_PIL_range(conditions, self.condition_max, self.condition_min)
        ssim, _, _, _, mae = calculate_metrics(generations_value, targets_value)
        save_images_range(target_images=targets_pixel, target_max=targets_max, target_min=targets_min,
                        generation_images=generations_pixel, generation_max=generations_max, generation_min=generations_min,
                        condition_images=conditions_pixel, condition_max=conditions_max, condition_min=conditions_min,
                        path=os.path.join(self.train_path, f"epoch_{epoch+1}.jpg"))
        # save_images(target_images=targets_pixel, output_images=samples_pixel, condition_images=conditions_pixel,
        #             path=os.path.join(self.train_path, f"epoch_pixel_{epoch+1}.jpg"))
        return np.mean(ssim), np.mean(mae)

    def update_ema(self):
        self.model.eval()
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def update_best_model(self, epoch):
        self.best_model_checkpoint = self.model.state_dict()
        self.best_model_epoch = epoch
        torch.save(self.best_model_checkpoint, os.path.join(self.model_path, "best_model.pth"))
       
    def train(self):
        logging.info(f"Training ...")
        self.model.to(self.device)
        if self.diffusion.conditioned_prior == True:
            self.diffusion.init_prior_mean_variance(self.train_dataloader, self.model_path)
            logging.info("Initialize prior mean and variance")

        if self.threshold_training == False:
            for epoch in range(self.epochs):

                self.train_epoch()
                val_loss = self.validation_epoch()

                logging.info(f"Epoch: {epoch + 1} Train Loss: {self.train_losses[-1]:.4f} Val Loss: {val_loss:.4f}")
                save_loss_image(train_loss=self.train_losses, val_loss=self.val_losses, path=os.path.join(self.train_path, "losses.png"))

                if val_loss < self.best_val_loss and self.ema == True:
                    self.best_val_loss = val_loss
                    self.update_best_model(epoch)

                if (epoch + 1) % self.sample_epoch == 0:
                    ssim, mae = self.generate_samples(epoch)
                    logging.info(f"Epoch: {epoch + 1} SSIM: {ssim:.2f} MAE: {mae:.2f}")
                    if self.ema == False and self.loss == 'l2' and mae < self.best_val_loss:
                        self.best_val_loss = mae
                        self.update_best_model(epoch)

                    self.ssim_values.append(ssim)
                    self.mae_values.append(mae)

        elif self.threshold_training == True:
            mae = float('inf')
            epoch = 0

            while mae > self.threshold:
                logging.info(f"Epoch {epoch}:")

                logging.info("Starting train loop")
                self.train_epoch()

                logging.info("Starting validation loop")
                val_loss = self.validation_epoch()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_checkpoint = self.model.state_dict()
                    self.best_model_epoch = epoch

                if (epoch + 1) % self.sample_epoch == 0:
                    ssim, mae = self.generate_samples(epoch)
                epoch += 1