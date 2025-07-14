import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
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

def save_images(target_images: list[Image]=None, generation_images: list[Image]=None, condition_images: list[Image]=None, path: str=None, **kwargs):
    """
    Save images in a grid format with titles for each row.

    Args:
        target_images (list[Image]): List of target images.
        generation_images (list[Image]): List of generation images.
        condition_images (list[Image]): List of condition images.
        path (str): Path to save the image grid.
        **kwargs: Additional keyword arguments for plt.savefig.
    """

    image_sets_with_titles = {
        'Target': target_images,
        'Generation': generation_images,
        'Condition': condition_images
    }

    image_sets = [(title, images) for title, images in image_sets_with_titles.items() if images is not None]

    n_rows = len(image_sets)
    n_cols = max(len(images) for _, images in image_sets)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

    for row, (title, image_set) in enumerate(image_sets):
        for col, img in enumerate(image_set):
            masked_img = image_add_mask(img)
            axs[row, col].imshow(masked_img, cmap='jet', vmin=0, vmax=255)
            
            axs[row, col].axis('off')
            if col == 0:
                axs[row, col].set_title(title, fontweight='bold', size=22, loc = 'left')

        for extra_col in range(col + 1, n_cols):
            axs[row, extra_col].axis('off')

    plt.tight_layout()
    if path:
        plt.savefig(path, **kwargs) 
    plt.close()

def save_images_range(target_images: list[Image]=None, target_max: list=None, target_min: list=None,
                    generation_images: list[Image]=None, generation_max: list=None, generation_min: list=None,
                    condition_images: list[Image]=None, condition_max: list=None, condition_min: list=None,
                    path: str=None, **kwargs):
    """
    Save images in a grid format with titles for each row, including max and min values.

    Args:
        target_images (list[Image]): List of target images.
        target_max (list): List of maximum values for target images.
        target_min (list): List of minimum values for target images.
        generation_images (list[Image]): List of generation images.
        generation_max (list): List of maximum values for generation images.
        generation_min (list): List of minimum values for generation images.
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
    
    if generation_images is not None:
        image_data.append({
            'title': 'Generation',
            'images': generation_images,
            'max_values': generation_max or [None] * len(generation_images),
            'min_values': generation_min or [None] * len(generation_images)
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
            # masked_img = image_add_mask(img)

            img_max = max_values[col] 
            img_min = min_values[col] 

            axs[row, col].imshow(img, cmap='jet', vmin=0, vmax=255)
            axs[row, col].axis('off')

            if col == 0:
                axs[row, col].set_title(title, fontweight='bold', size=12, loc='left')

            range_text = f'Max: {img_max:.2f}\nMin: {img_min:.2f}\n'
            axs[row, col].text(0.5, -0.1, range_text,
                            transform=axs[row, col].transAxes,
                            ha='center', va='top',
                            fontsize=10)
            
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

    plt.legend(fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Losses', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(path)
    plt.close()

def save_line_chart(data: list[float], sample_epoch: list[int], title: str, path: str):
    x_values = range(sample_epoch, (len(data) + 1) * sample_epoch, sample_epoch)
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, data, label=title, marker='o', linestyle='-')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

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
            image_tensor = image_tensor.squeeze(0)
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
            image_tensor = image_tensor.squeeze(0)
        image = transforms.ToPILImage()(image_tensor)
        value_images.append(image)

    temp = tensor * (max_value - min_value) + min_value

    batch_size, channels, height, width = temp.shape
    circular_mask = create_circular_mask(height, width).to(temp.device)
    circular_mask = circular_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, -1, -1)
    
    temp_masked = temp.clone()
    temp_masked[~circular_mask] = float('nan') 
    temp_masked_flat = temp_masked.view(batch_size, channels, -1)
    
    temp_masked_for_max = temp_masked_flat.clone()
    temp_masked_for_max[torch.isnan(temp_masked_for_max)] = float('-inf')
    temp_max = torch.max(temp_masked_for_max, dim=2, keepdim=True)[0]
    
    temp_masked_for_min = temp_masked_flat.clone()
    temp_masked_for_min[torch.isnan(temp_masked_for_min)] = float('inf')
    temp_min = torch.min(temp_masked_for_min, dim=2, keepdim=True)[0]

    temp_max = temp_max.unsqueeze(3)
    temp_min = temp_min.unsqueeze(3)
    
    pixels = (temp - temp_min) / (temp_max - temp_min)

    pixels = (pixels * 255).type(torch.uint8)
    pixel_images = []
    for i in range(tensor.shape[0]):
        image_tensor = pixels[i]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        image = transforms.ToPILImage()(image_tensor)
        pixel_images.append(image)
    temp_max_list = temp_max.squeeze().cpu().numpy().tolist()
    temp_min_list = temp_min.squeeze().cpu().numpy().tolist()

    value_images = [image_add_mask(img) for img in value_images]
    pixel_images = [image_add_mask(img) for img in pixel_images]
    return value_images, pixel_images, temp_max_list, temp_min_list

def image_add_mask(image: Image) -> Image:
    """
    Add a circular mask to the image to create a circular effect.

    Args:
        image: The original image to add the mask.
    """
    image_size = image.size[0]
    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((3, 3, image_size - 3, image_size - 3), fill=255)

    image_array = np.array(image)
    mask_array = np.array(mask)
    
    image_array[mask_array == 0] = 0

    # masked_img = np.ma.masked_where(~np.array(mask), np.array(image))
    return Image.fromarray(image_array)

def create_circular_mask(height, width, center=None, radius=None):
    """
    Create a circular mask for the given dimensions.
    
    Args:
        height: Height of the mask
        width: Width of the mask
        center: Center of the circle (y, x). If None, use image center.
        radius: Radius of the circle. If None, use minimum of height/width divided by 2.
    
    Returns:
        torch.Tensor: Boolean mask where True indicates inside the circle
    """
    if center is None:
        center = (height // 2, width // 2)
    if radius is None:
        radius = min(height, width) // 2 - 3
    
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    return mask

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