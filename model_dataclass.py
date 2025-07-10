import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image


def extract_label(filename: str):
    pattern = r'F_(?P<F>-?\d+(?:\.\d+)?)_dx_(?P<dx>-?\d+(?:\.\d+)?)_I_(?P<I>-?\d+(?:\.\d+)?)_angle_(?P<angle>-?\d+(?:\.\d+)?)_Tmax_(?P<Tmax>-?\d+(?:\.\d+)?)_Tmin_(?P<Tmin>-?\d+(?:\.\d+)?)'
    match = re.search(pattern, filename)
    if match:
        F = float(match.group('F'))
        dx = float(match.group('dx'))
        I = float(match.group('I'))
        angle = float(match.group('angle'))
        Tmax = float(match.group('Tmax'))
        Tmin = float(match.group('Tmin'))
        return F, dx, I, angle, Tmax, Tmin
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")

class LabeledDataset(Dataset):
    def __init__(self, dataset_dir:str, target_dir: str, condition_dir: str, result_dir: str, physics_informed: bool, data_transform=None):
        """
        Args:
            dataset_dir: Directory with range csv.
            target_dir: Directory with all the target images.
            condition_dir: Directory with all the condition images.
            result_dir: Directory to save the dataset range csv.
            transform: Optional transform to be applied on a sample.
        """
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.physics_informed = physics_informed
        self.data_transform = data_transform

        self.target_images = sorted(os.listdir(target_dir))
        self.condition_images = sorted(os.listdir(condition_dir))

        if not (len(self.target_images) == len(self.condition_images)):
            raise ValueError(
                f"Number of images in all directories must match. "
                f"Target: {len(self.target_images)}, "
                f"Condition: {len(self.condition_images)}"
            )
        
        t_range = pd.read_csv(os.path.join(dataset_dir, "T_range.csv"))
        self.surface_max = t_range["surface_max (°C)"].max()
        self.surface_min = t_range["surface_min (°C)"].min()
        self.side_max = t_range["side_max (°C)"].max()
        self.side_min = t_range["side_min (°C)"].min()
        self.condition_max = t_range["analysis_max (°C)"].max()
        self.condition_min = t_range["analysis_min (°C)"].min()
        self.gap_max = t_range["gap_max (°C)"].max()
        self.gap_min = t_range["gap_min (°C)"].min()
        # print(f"Surface temperature range: {self.surface_min}°C to {self.surface_max}°C")
        # print(f"Side temperature range: {self.side_min}°C to {self.side_max}°C")
        # print(f"Condition temperature range: {self.condition_min}°C to {self.condition_max}°C")
        # print(f"Gap temperature range: {self.gap_min}°C to {self.gap_max}°C")
        
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, "dataset_range.csv")
            pd.DataFrame({
                "surface_max": [self.surface_max],
                "surface_min": [self.surface_min],
                "side_max": [self.side_max],
                "side_min": [self.side_min],
                "condition_max": [self.condition_max],
                "condition_min": [self.condition_min],
                "gap_max": [self.gap_max],
                "gap_min": [self.gap_min]
            }).to_csv(result_path, index=False)

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        condition_image_path = os.path.join(self.condition_dir, self.condition_images[idx])

        mask = (read_image(target_image_path)[1:, ...] == 255).float()
        target_image = read_image(target_image_path)[:1, ...] * mask 
        condition_image = read_image(condition_image_path)[:1:, ...] * mask

        if self.data_transform:
            t_image = self.data_transform(target_image)
            c_image = self.data_transform(condition_image)

        t_F, t_dx, t_I, t_angle, t_Tmax, t_Tmin = extract_label(self.target_images[idx])
        c_F, c_dx, c_I, c_angle, c_Tmax, c_Tmin = extract_label(self.condition_images[idx])
        if not (t_F == c_F) or not (t_dx == c_dx) or not (t_I == c_I) or not (t_angle == c_angle):
            raise ValueError(
                f"Parameters must match across all three images for index {idx}:\n"
                f"  F values - Target: {t_F}, Condition: {c_F}\n"
                f"  dx values - Target: {t_dx}, Condition: {c_dx}\n"
                f"  I values - Target: {t_I}, Condition: {c_I}\n"
                f"  Angle values - Target: {t_angle}, Condition: {c_angle}"
            )

        t_image = t_image * (t_Tmax - t_Tmin) + t_Tmin
        c_image = c_image * (c_Tmax - c_Tmin) + c_Tmin

        if self.physics_informed:
            t_image = t_image - c_image
            t_image = (t_image - self.gap_min) / (self.gap_max - self.gap_min)
        else:
            t_image = (t_image - self.surface_min) / (self.surface_max - self.surface_min)

        c_image = (c_image - self.condition_min) / (self.condition_max - self.condition_min)

        t_image = ((t_image * 2) - 1) 
        c_image = ((c_image * 2) - 1)

        # t_image, _, _ = self.Cartesian2Polar(t_image, center=None, max_radius=t_image.shape[2] // 2 - 2)
        # c_image, _, _ = self.Cartesian2Polar(c_image, center=None, max_radius=c_image.shape [2] // 2 - 2)

        return t_image, c_image
    
    @staticmethod
    def Cartesian2Polar(img_tensor, center=None, max_radius=None):
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0) 
        
        B, C, H, W = img_tensor.shape

        img_max = img_tensor.view(B, -1).max(dim=1)[0]
        img_min = img_tensor.view(B, -1).min(dim=1)[0]
        
        if center is None:
            center = (W // 2, H // 2)
        if max_radius is None:
            max_radius = min(center[0], center[1])
        
        theta = torch.linspace(0, 2 * np.pi, W, device=img_tensor.device)
        r = torch.linspace(0, max_radius, H, device=img_tensor.device)
        
        theta_grid, r_grid = torch.meshgrid(theta, r, indexing='ij')
        theta_grid = theta_grid.T 
        r_grid = r_grid.T
        
        x = center[0] + r_grid * torch.cos(theta_grid)
        y = center[1] + r_grid * torch.sin(theta_grid)

        x = 2 * x / (W - 1) - 1
        y = 2 * y / (H - 1) - 1
        
        grid = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        polar_img = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return polar_img.squeeze(0) if polar_img.shape[0] == 1 else polar_img, img_max, img_min

    @staticmethod
    def Polar2Cartesian(polar_tensor, center=None, max_radius=None, polar_max=None, polar_min=None):
        if polar_tensor.dim() == 3:
            polar_tensor = polar_tensor.unsqueeze(0) 
        
        B, C, H, W = polar_tensor.shape
        
        if center is None:
            center = (W // 2, H // 2)
        if max_radius is None:
            max_radius = min(center[0], center[1])
        
        x = torch.linspace(0, W - 1, W, device=polar_tensor.device)
        y = torch.linspace(0, H - 1, H, device=polar_tensor.device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        x_grid = x_grid.T
        y_grid = y_grid.T
        
        dx = x_grid - center[0]
        dy = y_grid - center[1]
        
        r = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        
        r_coord = r * (H - 1) / max_radius
        theta_coord = theta * (W - 1) / (2 * np.pi)
        
        r_coord_norm = 2 * r_coord / (H - 1) - 1
        theta_coord_norm = 2 * theta_coord / (W - 1) - 1
        r_coord_norm = torch.clamp(r_coord_norm, -1, 1)
        theta_coord_norm = torch.clamp(theta_coord_norm, -1, 1)
        
        grid = torch.stack([theta_coord_norm, r_coord_norm], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        cartesian_img = F.grid_sample(polar_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        valid_mask = r <= max_radius
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
        fill_value = torch.full_like(cartesian_img, -1.0)
        cartesian_img = torch.where(valid_mask, cartesian_img, fill_value)
        
        if polar_max is not None or polar_min is not None:
            for idx in range(cartesian_img.shape[0]):
                cartesian_img[idx] = torch.clamp(cartesian_img[idx], min=polar_min[idx].item(), max=polar_max[idx].item())
                cartesian_img[idx] = (cartesian_img[idx] - cartesian_img[idx].min()) / (cartesian_img[idx].max() - cartesian_img[idx].min() + 1e-8)
                cartesian_img[idx] = cartesian_img[idx] * (polar_max[idx] - polar_min[idx]) + polar_min[idx]
        
        return cartesian_img.squeeze(0) if cartesian_img.shape[0] == 1 else cartesian_img     