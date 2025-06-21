import os
import re
import torch
import pandas as pd

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
    def __init__(self, dataset_dir:str, target_dir: str, condition_dir: str, analysis_dir: str, result_dir: str, physics_informed: bool, data_transform=None):
        """
        Args:
            dataset_dir: Directory with range csv.
            target_dir: Directory with all the target images.
            condition_dir: Directory with all the condition images.
            analysis_dir: Directory with all the analysis images.
            result_dir: Directory to save the dataset range csv.
            transform: Optional transform to be applied on a sample.
        """
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.analysis_dir = analysis_dir
        self.physics_informed = physics_informed
        self.data_transform = data_transform

        self.target_images = sorted(os.listdir(target_dir))
        self.condition_images = sorted(os.listdir(condition_dir))
        self.analysis_images = sorted(os.listdir(analysis_dir))

        if not (len(self.target_images) == len(self.condition_images) == len(self.analysis_images)):
            raise ValueError(
                f"Number of images in all directories must match. "
                f"Target: {len(self.target_images)}, "
                f"Condition: {len(self.condition_images)}, "
                f"Analysis: {len(self.analysis_images)}"
            )
        
        t_range = pd.read_csv(os.path.join(dataset_dir, "T_range.csv"))
        self.surface_max = t_range["surface_max (°C)"].max()
        self.surface_min = t_range["surface_min (°C)"].min()
        self.side_max = t_range["side_max (°C)"].max()
        self.side_min = t_range["side_min (°C)"].min()
        self.analysis_max = t_range["analysis_max (°C)"].max()
        self.analysis_min = t_range["analysis_min (°C)"].min()
        self.gap_max = t_range["gap_max (°C)"].max()
        self.gap_min = t_range["gap_min (°C)"].min()
        print(f"Surface temperature range: {self.surface_min}°C to {self.surface_max}°C")
        print(f"Side temperature range: {self.side_min}°C to {self.side_max}°C")
        print(f"Analysis temperature range: {self.analysis_min}°C to {self.analysis_max}°C")
        print(f"Gap temperature range: {self.gap_min}°C to {self.gap_max}°C")
        
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, "dataset_range.csv")
            pd.DataFrame({
                "surface_max": [self.surface_max],
                "surface_min": [self.surface_min],
                "side_max": [self.side_max],
                "side_min": [self.side_min],
                "analysis_max": [self.analysis_max],
                "analysis_min": [self.analysis_min],
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
        analysis_image_path = os.path.join(self.analysis_dir, self.analysis_images[idx])

        mask = (read_image(target_image_path)[1:, ...] == 255).float()
        target_image = read_image(target_image_path)[:1, ...] * mask 
        condition_image = read_image(condition_image_path)[:1:, ...] * mask
        analysis_image = read_image(analysis_image_path)[:1:, ...]  * mask

        if self.data_transform:
            t_image = self.data_transform(target_image)
            c_image = self.data_transform(condition_image)
            a_image = self.data_transform(analysis_image)

        t_F, t_dx, t_I, t_angle, t_Tmax, t_Tmin = extract_label(self.target_images[idx])
        c_F, c_dx, c_I, c_angle, c_Tmax, c_Tmin = extract_label(self.condition_images[idx])
        a_F, a_dx, a_I, a_angle, a_Tmax, a_Tmin = extract_label(self.analysis_images[idx])
        if not (t_F == c_F == a_F) or not (t_dx == c_dx == a_dx) or not (t_I == c_I == a_I) or not (t_angle == c_angle == a_angle):
            raise ValueError(
                f"Parameters must match across all three images for index {idx}:\n"
                f"  F values - Target: {t_F}, Condition: {c_F}, Analysis: {a_F}\n"
                f"  dx values - Target: {t_dx}, Condition: {c_dx}, Analysis: {a_dx}\n"
                f"  I values - Target: {t_I}, Condition: {c_I}, Analysis: {a_I}\n"
                f"  angle values - Target: {t_angle}, Condition: {c_angle}, Analysis: {a_angle}"
            )

        t_image = t_image * (t_Tmax - t_Tmin) + t_Tmin
        c_image = c_image * (c_Tmax - c_Tmin) + c_Tmin
        a_image = a_image * (a_Tmax - a_Tmin) + a_Tmin

        if self.physics_informed:
            t_image = t_image - a_image
            t_image = (t_image - self.gap_min) / (self.gap_max - self.gap_min)
        else:
            t_image = (t_image - self.surface_min) / (self.surface_max - self.surface_min)
        c_image = (c_image - self.side_min) / (self.side_max - self.side_min)
        a_image = (a_image - self.analysis_min) / (self.analysis_max - self.analysis_min)

        t_image = ((t_image * 2) - 1) 
        c_image = ((c_image * 2) - 1)
        a_image = ((a_image * 2) - 1)

        return t_image, c_image, a_image