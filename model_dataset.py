import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity as ssim
from model_utils import *

from model_dataclass import LabeledDataset


def normalize_softmax(x):
    min = x.min(0, keepdims=True)
    max = x.max(0, keepdims=True)
    x = (x - min) / (max - min)
    np.fill_diagonal(x, -np.inf)
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)

def SSIM_matrix(dataset1, dataset2):

    matrix = np.zeros((len(dataset1), len(dataset2)))

    for i, (input_image1, label_image1, _) in enumerate(dataset1):
        print(f"Calculation step: {i}")
        input_image1 = tensor_to_PIL(input_image1)

        for j, (input_image2, label_image2, _) in enumerate(dataset2):
            input_image2 = tensor_to_PIL((input_image2))
            matrix[i,j] = ssim(np.array(input_image1), np.array(input_image2), channel_axis=0, multichannel=True)

    return normalize_softmax(matrix)

def show_similarity_pair(dataset, similarity_vector, indices):

    for index in indices:
        image1, _, _ = dataset[index]
        image2, _, _ = dataset[similarity_vector[index]]

        image1, image2 = np.transpose(tensor_to_PIL(image1), (1, 2, 0)), np.transpose(tensor_to_PIL(image2), (1, 2, 0))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(image1)
        axs[1].imshow(image2)

        plt.show()

def save_ordered_dataset(dataset, indices, path):
    images = []

    for index in indices:
        image, _, _ = dataset[index]
        image = np.transpose(tensor_to_PIL(image), (1, 2, 0))
        image = Image.fromarray(image)
        images.append(image)

    save_image_list(images, path)

def get_data(dataset_path: str, target_dataset_path: str, condition_dataset_path: str, result_path: str = None, split: bool = True, **kwargs):
    loss = kwargs.get("loss")
    test_split = kwargs.get("test_split")
    validation_split = kwargs.get("validation_split")
    batch_size = kwargs.get("batch_size")
    resolution = kwargs.get("resolution")

    if loss == "l2":
        data_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
    elif loss == "vlb":
        data_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
        ])
    else:
        raise ValueError("Invalid loss function")
    
    # if loss == "l2":
    #     data_transform = transforms.Compose([
    #         transforms.Resize((resolution, resolution)),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.Lambda(lambda t: t / 255.0),
    #         transforms.Lambda(lambda t: (t * 2) - 1)
    #     ])
    # elif loss == "vlb":
    #     data_transform = transforms.Compose([
    #         transforms.Resize((resolution, resolution)),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.Lambda(lambda t: t / 255.0)
    #     ])
    # else:
    #     raise ValueError("Invalid loss function")

    dataset = LabeledDataset(dataset_path, target_dataset_path, condition_dataset_path, result_path,data_transform=data_transform)

    if split == True:
        train_size = int((1 - test_split - validation_split) * len(dataset))
        val_size = int(validation_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if result_path is not None:
            torch.save(train_dataset.indices, os.path.join(result_path, "train_indices.pth"))
            torch.save(val_dataset.indices, os.path.join(result_path, "val_indices.pth"))
            torch.save(test_dataset.indices, os.path.join(result_path, "test_indices.pth"))

        return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader, dataset, -1, -1, -1, -1