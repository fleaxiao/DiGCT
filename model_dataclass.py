import os
import re
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image


def extract_label(filename: str):
    pattern = r'F_(?P<F>[\d\-\.]+)_dx_(?P<dx>[\d\-\.]+)_I_(?P<I>[\d\-\.]+)_angle_(?P<angle>[\d\-\.]+)_Tmax_(?P<Tmax>[\d\.]+)_Tmin_(?P<Tmin>\d+(?:\.\d+)?)'
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
    def __init__(self, dataset_dir:str, target_dir: str, condition_dir: str, result_dir: str,data_transform=None):
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
        self.data_transform = data_transform

        self.target_images = sorted(os.listdir(target_dir))
        self.condition_images = sorted(os.listdir(condition_dir))

        t_range = pd.read_csv(os.path.join(dataset_dir, "T_range.csv"))
        self.surface_max = t_range["surface_max (°C)"].max()
        self.surface_min = t_range["surface_min (°C)"].min()
        self.side_max = t_range["side_max (°C)"].max()
        self.side_min = t_range["side_min (°C)"].min()
        print(f"Surface temperature range: {self.surface_min}°C to {self.surface_max}°C")
        print(f"Side temperature range: {self.side_min}°C to {self.side_max}°C")

        if len(self.target_images) != len(self.condition_images):
            raise ValueError("Number of target images and condition images do not match")
        
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, "dataset_range.csv")
            pd.DataFrame({
                "surface_max": [self.surface_max],
                "surface_min": [self.surface_min],
                "side_max": [self.side_max],
                "side_min": [self.side_min]
            }).to_csv(result_path, index=False)

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        condition_image_path = os.path.join(self.condition_dir, self.condition_images[idx])

        t_F, t_dx, t_I, t_angle, t_Tmax, t_Tmin = extract_label(self.target_images[idx])
        c_F, c_dx, c_I, c_angle, c_Tmax, c_Tmin = extract_label(self.condition_images[idx])
        if t_F != c_F or t_dx != c_dx or t_I != c_I or t_angle != c_angle:
            raise ValueError(f"Labels do not match for images: Surface {self.target_images[idx]} and Side {self.condition_images[idx]}")

        target_image = read_image(target_image_path)[:1, ...]
        condition_image = read_image(condition_image_path)[:1, ...]
        
        # if self.data_transform:
        #     t_image = self.data_transform(target_image)
        #     c_image = self.data_transform(condition_image)
        #     m_image = torch.ones_like(c_image)

        target_image = target_image / 255 * (t_Tmax - t_Tmin) + t_Tmin
        condition_image = condition_image / 255 * (c_Tmax - c_Tmin) + c_Tmin

        target_image = (target_image - self.surface_min) / (self.surface_max - self.surface_min)
        condition_image = (condition_image - self.side_min) / (self.side_max - self.side_min)

        if self.data_transform:
            t_image = self.data_transform(target_image)
            c_image = self.data_transform(condition_image)

        m_image = torch.ones_like(c_image)
        # print()
        # print(t_image.max(), t_image.min())
        # print(c_image.max(), c_image.min())

        return t_image, c_image, m_image