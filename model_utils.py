import os
import torch
import pandas as pd
import torchvision
import torch.nn as nn
import re
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


def load_images(folder_path: str):
    images = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(re.search(r'\d+', x).group())):
        print(filename)
        img = Image.open(os.path.join(folder_path, filename))
        images.append(img)
    return images

def save_image_list(image_list: list[Image], path: str):
    for i, image in enumerate(image_list):
        image.save(os.path.join(path, f"{i}.png"))

def save_images(reference_images: list[Image]=None, generated_images: list[Image]=None, condition_images: list[Image]=None, path: str=None, **kwargs):
    # Determine how many image sets are provided
    image_sets_with_titles = {
        'Reference': reference_images,
        'Generated': generated_images,
        'Condition': condition_images
    }

    image_sets = [(title, images) for title, images in image_sets_with_titles.items() if images is not None]
    # Calculate the maximum number of images in any set to set the number of columns
    n_rows = len(image_sets)
    n_cols = max(len(images) for _, images in image_sets)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

    for row, (title, image_set) in enumerate(image_sets):
        for col, img in enumerate(image_set):
            axs[row, col].imshow(img, cmap='gray')  # Display the image
            axs[row, col].axis('off')  # Hide the axes ticks
            if col == 0:
                axs[row, col].set_title(title, fontweight='bold', size=20, loc = 'left')  # Set the title of the row

            # Fill remaining columns of the row with blank axes if any
            for extra_col in range(col + 1, n_cols):
                axs[row, extra_col].axis('off')

    plt.tight_layout()
    if path:
        plt.savefig(path, **kwargs)  # Save the figure if a path is provided
    plt.close()  # Close the figure to free memory

def save_loss_image(train_loss: list[float], val_loss: list[float], path: str):
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Loss over Epochs')
    plt.savefig(path)
    plt.close()

def concatenate_images(images: torch.Tensor, structures: torch.Tensor):
    return torch.cat((images, structures), dim=1)

def split_images(concatenated_images: torch.Tensor):
    return concatenated_images[:, :3], concatenated_images[:, 3:]

def concat_to_batchsize(images: torch.Tensor, n: int):
    m = images.shape[0]
    if m == n:
        return images
    elif m < n:
        indices = torch.arange(0, n-m)
        return torch.cat((images, images[indices]), dim=0)
    else:
        return images[:n]

def tensor_to_PIL(tensor: torch.Tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).type(torch.uint8)
    images = []
    for i in range(tensor.shape[0]):
        image_tensor = tensor[i]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)  # Remove the batch dimension
        image = transforms.ToPILImage()(image_tensor)
        images.append(image)
    return images

def convert_grey_to_white(image: Image, threshold: int = 200):
    """
    Converts light gray pixels in the image to white.

    Parameters:
    - image: The Image to be Converted
    - threshold: The value above which a pixel will be set to white.
    """

    image_array = np.array(image)
    # Define a function to apply to each pixel
    def change_color(pixel):
        # If all the channels of the pixel value are above the threshold, change it to white
        if np.all(pixel > threshold):
            return (255, 255, 255)
        else:
            return pixel

    # Apply the function to each pixel in the image
    new_image_array = np.apply_along_axis(change_color, axis=-1, arr=image_array)
    new_image = Image.fromarray(new_image_array.astype('uint8'), 'RGB')
    return new_image

def convert_black_to_white(image: Image):
    image_array = np.array(image)

    # Define a function to apply to each pixel
    def change_color(pixel):
        # If all the channels of the pixel value are above the threshold, change it to white
        if np.all(pixel < 5):
            return (255, 255, 255)
        else:
            return pixel

    # Apply the function to each pixel in the image
    new_image_array = np.apply_along_axis(change_color, axis=-1, arr=image_array)
    new_image = Image.fromarray(new_image_array.astype('uint8'), 'RGB')
    return new_image

def set_seed(seed: int, fully_deterministic: bool = False):
    """
    Set seed for reproducible behavior.

    Parameters
    ----------
    seed : int
        Seed value to set. By default, 1958.
    fully_deterministic : bool
        Whether to set the environment to fully deterministic. By default, False.
        This should only be used for debugging and testing, as it can significantly
        slow down training at little to no benefit.
    """
    if fully_deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Experiment seed set")

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def unsqueeze_right(x, num_dims=1):
    """
    Unsqueezes the last `num_dims` dimensions of `x`.
    """
    return x.view(x.shape + (1,) * num_dims)