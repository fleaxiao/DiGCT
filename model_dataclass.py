import os
import re
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image


def extract_label_from_filename(filename: str):
    pattern = r'^[^_]+_(\d{8})_(\d{6})_([^_]+)_\d\.png$'
    match = re.search(pattern, filename)
    if match:
        date, time, label = match.group(1), match.group(2), match.group(3)
        return label
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")

def extract_labels(target_dir: str):
    files = os.listdir(target_dir)
    label_dict = {}
    for i, file in enumerate(files):
        dimensions = extract_label_from_filename(file)
        label_dict[file] = dimensions
        label_dict[file]["index"] = i

    return label_dict

class LabeledDataset(Dataset):
    def __init__(self, target_dir: str, condition_dir: str, data_transform=None):
        """
        Args:
            target_dir: Directory with all the input images.
            condition_dir: Directory with all the label images.
            transform: Optional transform to be applied on a sample.
        """
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.data_transform = data_transform

        self.input_images = sorted(os.listdir(target_dir))
        self.label_images = self.input_images

        if len(self.input_images) != len(self.label_images):
            raise ValueError("Number of input images and label images do not match")

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_image_path = os.path.join(self.target_dir, self.input_images[idx])
        condition_image_path = os.path.join(self.condition_dir, self.label_images[idx])
        
        target_image = read_image(target_image_path)
        condition_image = read_image(condition_image_path)


        if self.data_transform:
            t_image = self.data_transform(target_image)
            c_image = self.data_transform(condition_image)
            m_image = torch.ones_like(c_image)

        return t_image, c_image, m_image