import os
import torch
import pandas as pd
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

def save_images(target_images: list[Image]=None, output_images: list[Image]=None, condition_images: list[Image]=None, path: str=None, **kwargs):
    """
    Save images in a grid format with titles for each row.

    Args:
        target_images (list[Image]): List of target images.
        output_images (list[Image]): List of output images.
        condition_images (list[Image]): List of condition images.
        path (str): Path to save the image grid.
        **kwargs: Additional keyword arguments for plt.savefig.
    """

    image_sets_with_titles = {
        'Target': target_images,
        'Output': output_images,
        'Condition': condition_images
    }

    image_sets = [(title, images) for title, images in image_sets_with_titles.items() if images is not None]

    n_rows = len(image_sets)
    n_cols = max(len(images) for _, images in image_sets)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

    for row, (title, image_set) in enumerate(image_sets):
        for col, img in enumerate(image_set):
            masked_img = convert_masked_image(img)
            axs[row, col].imshow(masked_img, cmap='jet', vmin=0, vmax=255)
            
            axs[row, col].axis('off')
            if col == 0:
                axs[row, col].set_title(title, fontweight='bold', size=25, loc = 'left', fontname='Times New Roman')

        for extra_col in range(col + 1, n_cols):
            axs[row, extra_col].axis('off')

    plt.tight_layout()
    if path:
        plt.savefig(path, **kwargs) 
    plt.close()

def save_images_range(target_images: list[Image]=None, target_max: list=None, target_min: list=None,
                    output_images: list[Image]=None, output_max: list=None, output_min: list=None,
                    condition_images: list[Image]=None, condition_max: list=None, condition_min: list=None,
                    path: str=None, **kwargs):
    """
    Save images in a grid format with titles for each row, including max and min values.

    Args:
        target_images (list[Image]): List of target images.
        target_max (list): List of maximum values for target images.
        target_min (list): List of minimum values for target images.
        output_images (list[Image]): List of output images.
        output_max (list): List of maximum values for output images.
        output_min (list): List of minimum values for output images.
        condition_images (list[Image]): List of condition images.
        condition_max (list): List of maximum values for condition images.
        condition_min (list): List of minimum values for condition images.
        path (str): Path to save the image grid.
        **kwargs: Additional keyword arguments for plt.savefig.
    """

    image_data = []
    
    if target_images is not None:
        image_data.append({
            'title': 'Target',
            'images': target_images,
            'max_values': target_max or [None] * len(target_images),
            'min_values': target_min or [None] * len(target_images)
        })
    
    if output_images is not None:
        image_data.append({
            'title': 'Output',
            'images': output_images,
            'max_values': output_max or [None] * len(output_images),
            'min_values': output_min or [None] * len(output_images)
        })
    
    if condition_images is not None:
        image_data.append({
            'title': 'Condition',
            'images': condition_images,
            'max_values': condition_max or [None] * len(condition_images),
            'min_values': condition_min or [None] * len(condition_images)
        })

    n_rows = len(image_data)
    n_cols = max(len(data['images']) for data in image_data)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 3), squeeze=False)

    for row, data in enumerate(image_data):
        title = data['title']
        images = data['images']
        max_values = data['max_values']
        min_values = data['min_values']

        for col, img in enumerate(images):
            masked_img = convert_masked_image(img)

            img_max = max_values[col] 
            img_min = min_values[col] 

            axs[row, col].imshow(masked_img, cmap='jet', vmin=0, vmax=255)
            axs[row, col].axis('off')

            if col == 0:
                axs[row, col].set_title(title, fontweight='bold', size=15, loc='left', fontname='Times New Roman')

            range_text = f'Max: {img_max:.2f}\nMin: {img_min:.2f}\n'
            axs[row, col].text(0.5, -0.1, range_text,
                            transform=axs[row, col].transAxes,
                            ha='center', va='top',
                            fontsize=10,
                            fontname='Times New Roman')
            
        for extra_col in range(len(images), n_cols):
            axs[row, extra_col].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if path:
        plt.savefig(path, bbox_inches='tight', **kwargs)
    plt.close()

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

def tensor_to_PIL_range(tensor: torch.Tensor, max_value: float, min_value: float):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = tensor.type(torch.float32)

    values = (tensor * 255).type(torch.uint8)
    value_images = []
    for i in range(tensor.shape[0]):
        image_tensor = values[i]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)  # Remove the batch dimension
        image = transforms.ToPILImage()(image_tensor)
        value_images.append(image)

    temp = tensor * (max_value - min_value) + min_value
    temp_max = torch.amax(temp, dim=(1, 2, 3), keepdim=True) 
    temp_min = torch.amin(temp, dim=(1, 2, 3), keepdim=True) 
    pixels = (temp - temp_min) / (temp_max - temp_min)

    pixels = (pixels * 255).type(torch.uint8)
    pixel_images = []
    for i in range(tensor.shape[0]):
        image_tensor = pixels[i]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)  # Remove the batch dimension
        image = transforms.ToPILImage()(image_tensor)
        pixel_images.append(image)
    temp_max_list = temp_max.squeeze().cpu().numpy().tolist()
    temp_min_list = temp_min.squeeze().cpu().numpy().tolist()
    return value_images, pixel_images, temp_max_list, temp_min_list

def convert_masked_image(image: Image):
    """
    Converts an image to a masked image with a circular mask.

    Parameters:
    - image: The Image to be Converted

    Returns:
    - masked_img: The masked image with a circular mask applied
    """
    img_array = np.array(image)

    h, w = img_array.shape

    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 2 - 1

    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    img_rgba = np.zeros((h, w, 2), dtype=img_array.dtype)
    img_rgba[:, :, 0] = img_array 
    img_rgba[:, :, 1] = 255 
    img_rgba[~mask, 1] = 0
    
    masked_img = np.ma.masked_where(~mask, img_array)
    return masked_img

def convert_grey_to_white(image: Image, threshold: int = 200):
    """
    Converts light gray pixels in the image to white.

    Parameters:
    - image: The Image to be Converted
    - threshold: The value above which a pixel will be set to white
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