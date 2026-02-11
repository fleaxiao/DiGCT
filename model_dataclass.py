import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image


def extract_label(filename: str):
    pattern = r'F_(?P<F>-?\d+(?:\.\d+)?)_theta_(?P<theta>-?\d+(?:\.\d+)?)_vf_(?P<vf>-?\d+(?:\.\d+)?)_I_(?P<I>-?\d+(?:\.\d+)?)_angle_(?P<angle>-?\d+(?:\.\d+)?)_Tmax_(?P<Tmax>-?\d+(?:\.\d+)?)_Tmin_(?P<Tmin>-?\d+(?:\.\d+)?)'
    # pattern = r'F_(?P<F>-?\d+(?:\.\d+)?)_dx_(?P<dx>-?\d+(?:\.\d+)?)_I_(?P<I>-?\d+(?:\.\d+)?)_angle_(?P<angle>-?\d+(?:\.\d+)?)_Tmax_(?P<Tmax>-?\d+(?:\.\d+)?)_Tmin_(?P<Tmin>-?\d+(?:\.\d+)?)'
    match = re.search(pattern, filename)
    if match:
        F = float(match.group('F'))
        theta = float(match.group('theta'))
        vf = float(match.group('vf'))
        I = float(match.group('I'))
        angle = float(match.group('angle'))
        Tmax = float(match.group('Tmax'))
        Tmin = float(match.group('Tmin'))
        return F, theta, vf, I, angle, Tmax, Tmin
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")

class LabeledDataset(Dataset):
    def __init__(self, dataset_dir:str, target_dir: str, condition_dir: str, result_dir: str, physics_constraint: bool, data_transform=None):
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
        self.physics_constraint = physics_constraint
        self.data_transform = data_transform

        self.target_images = sorted(os.listdir(target_dir))
        self.condition_images = sorted(os.listdir(condition_dir))

        if not (len(self.target_images) == len(self.condition_images)):
            raise ValueError(
                f"Number of images in all directories must match. "
                f"Target: {len(self.target_images)}, "
                f"Condition: {len(self.condition_images)}"
            )
        
        range_path = os.path.join(result_dir, "dataset_range.csv")
        if os.path.exists(range_path):
            range_df = pd.read_csv(range_path)
            self.surface_max = range_df["surface_max"].values[0]
            self.surface_min = range_df["surface_min"].values[0]
            self.side_max = range_df["side_max"].values[0]
            self.side_min = range_df["side_min"].values[0]
            self.condition_max = range_df["condition_max"].values[0]
            self.condition_min = range_df["condition_min"].values[0]
            self.gap_max = range_df["gap_max"].values[0]
            self.gap_min = range_df["gap_min"].values[0]
        else:
            t_range = pd.read_csv(os.path.join(dataset_dir, "T_range.csv"))
            self.surface_max = t_range["surface_max (°C)"].max()
            self.surface_min = t_range["surface_min (°C)"].min()
            self.side_max = t_range["side_max (°C)"].max()
            self.side_min = t_range["side_min (°C)"].min()
            self.condition_max = t_range["analysis_max (°C)"].max()
            self.condition_min = t_range["analysis_min (°C)"].min()
            self.gap_max = t_range["gap_max (°C)"].max()
            self.gap_min = t_range["gap_min (°C)"].min()

        os.makedirs(result_dir, exist_ok=True)
        pd.DataFrame({
        "surface_max": [self.surface_max],
        "surface_min": [self.surface_min],
        "side_max": [self.side_max],
        "side_min": [self.side_min],
        "condition_max": [self.condition_max],
        "condition_min": [self.condition_min],
        "gap_max": [self.gap_max],
        "gap_min": [self.gap_min]
        }).to_csv(range_path, index=False)

        print(f"Surface temperature range: {self.surface_min}°C to {self.surface_max}°C")
        print(f"Side temperature range: {self.side_min}°C to {self.side_max}°C")
        print(f"Condition temperature range: {self.condition_min}°C to {self.condition_max}°C")
        print(f"Gap temperature range: {self.gap_min}°C to {self.gap_max}°C")

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

        t_F, t_theta, t_vf, t_I, t_angle, t_Tmax, t_Tmin = extract_label(self.target_images[idx])
        c_F, c_theta, c_vf, c_I, c_angle, c_Tmax, c_Tmin = extract_label(self.condition_images[idx])
        if not (t_F == c_F) or not (t_theta == c_theta) or not (t_vf == c_vf) or not (t_I == c_I) or not (t_angle == c_angle):
            raise ValueError(
                f"Parameters must match across all three images for index {idx}:\n"
                f"  F values - Target: {t_F}, Condition: {c_F}\n"
                f"  theta values - Target: {t_theta}, Condition: {c_theta}\n"
                f"  vf values - Target: {t_vf}, Condition: {c_vf}\n"
                f"  I values - Target: {t_I}, Condition: {c_I}\n"
                f"  Angle values - Target: {t_angle}, Condition: {c_angle}"
            )

        t_image = t_image * (t_Tmax - t_Tmin) + t_Tmin
        c_image = c_image * (c_Tmax - c_Tmin) + c_Tmin

        if self.physics_constraint:
            t_image = t_image - c_image
            t_image = (t_image - self.gap_min) / (self.gap_max - self.gap_min)
        else:
            t_image = (t_image - self.surface_min) / (self.surface_max - self.surface_min)

        c_image = (c_image - self.condition_min) / (self.condition_max - self.condition_min)

        t_image = ((t_image * 2) - 1) 
        c_image = ((c_image * 2) - 1)

        return t_image, c_image, t_F, t_theta, t_vf, t_I, t_angle
