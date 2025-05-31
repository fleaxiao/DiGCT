import copy
import os
import torch
import time
import torch.nn as nn
import numpy as np
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

    def train_epoch(self):
        loss_total = 0
        self.model.train()
        pbar = tqdm(self.train_dataloader)

        for i, (images, conditions, masks) in enumerate(pbar):
            images, conditions, masks = images.to(self.device), conditions.to(self.device), masks.to(self.device)
            t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device) # l2: [0, 1000(noise step)], vlb: [0, 1)
            losses = self.diffusion.training_losses(model=self.model, x_start=images, c=conditions, m=masks, t=t)
            loss = losses["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            if self.ema == True:
                self.update_ema()

            loss_total += loss.item()
            pbar.set_postfix(MSE=loss.item())

        average_loss = loss_total / len(self.train_dataloader)
        self.train_losses.append(average_loss)

    def validation_epoch(self):
        loss_total = 0
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader)

            for i, (images, conditions, masks) in enumerate(pbar):
                images, conditions, masks = images.to(self.device), conditions.to(self.device), masks.to(self.device)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                losses = self.diffusion.training_losses(model=self.model, x_start=images, c=conditions, m=masks, t=t)
                loss = losses["loss"].mean()

                pbar.set_postfix(MSE=loss.item())
                loss_total += loss.item()

            average_loss = loss_total / len(self.val_dataloader)
            self.val_losses.append(average_loss)
            return average_loss

    def generate_samples(self, epoch):
        test_images, test_conditions, test_masks = next(cycle(self.test_dataloader))
        test_images = concat_to_batchsize(test_images, self.sample_number)
        test_conditions = test_conditions.to(self.device)
        test_masks = test_masks.to(self.device)
        test_conditions = concat_to_batchsize(test_conditions, self.sample_number)
        test_masks = concat_to_batchsize(test_masks, self.sample_number)
        
        if self.ema == True:
            sampled_images, condition_images = self.diffusion.p_sample_loop(self.ema_model, n=self.sample_number, c=test_conditions, m=test_masks, resolution=self.resolution)
        else:
            sampled_images, condition_images = self.diffusion.p_sample_loop(self.model, n=self.sample_number, c=test_conditions, m=test_masks, resolution=self.resolution)
        test_images = tensor_to_PIL(test_images)
        sampled_images = tensor_to_PIL(sampled_images)
        condition_images = tensor_to_PIL(condition_images)
        ssim, _, _, _, mae = calculate_metrics(sampled_images, test_images)
        save_images(reference_images=test_images, generated_images=sampled_images,
                    condition_images=condition_images, path=os.path.join(self.train_path, f"epoch_{epoch+1}.jpg"))
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
        logging.info(f"Starting training on {self.device}")
        self.model.to(self.device)
        if self.diffusion.conditioned_prior == True:
            self.diffusion.init_prior_mean_variance(self.train_dataloader, self.model_path)
            logging.info("Initialized prior mean and variance")
        start_time = time.time()
        if self.threshold_training == False:
            for epoch in range(self.epochs):
                logging.info(f"Starting epoch {epoch + 1}:")

                logging.info("Starting train loop")
                self.train_epoch()

                logging.info("Starting validation loop")
                val_loss = self.validation_epoch()

                save_loss_image(train_loss=self.train_losses, val_loss=self.val_losses, path=os.path.join(self.train_path, "training_losses.png"))

                if val_loss < self.best_val_loss and self.ema == True:
                    self.best_val_loss = val_loss
                    self.update_best_model(epoch)

                if (epoch + 1) % self.sample_epoch == 0:
                    ssim, mae = self.generate_samples(epoch)
                    if self.ema == False and self.loss == 'l2' and mae < self.best_val_loss:
                        self.best_val_loss = mae
                        self.update_best_model(epoch)

                    self.ssim_values.append(ssim)
                    self.mae_values.append(mae)

        elif self.threshold_training == True:
            mae = float('inf')
            epoch = 0

            while mae > self.threshold:
                logging.info(f"Starting epoch {epoch}:")

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

        end_time = time.time()
        logging.info(f"Training took {(end_time - start_time):.2f} seconds")