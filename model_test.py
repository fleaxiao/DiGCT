import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image, ImageColor

from torch.utils.data import DataLoader
from model_utils import *
from model_dataset import get_data
from model_diff import VDM_Tools, DDPM_Tools


def sample_model_output(
        model: torch.nn.Module,
        device: torch.device,
        sampler: DDPM_Tools,
        dataloader: DataLoader,
        resolution: int,
        length: int,
        batch_size: int,
):
    targets_list = []
    generated_list = []
    conditions_list = []
    iterator = iter(dataloader)

    if length < batch_size:
        raise ValueError("Length must be greater than batch size")
    
    else:
        for i in range(0, length, batch_size):
            targets, conditions, analyses = next(iterator)
            conditions = conditions.to(device)
            targets = targets.to(device)
            analyses = analyses.to(device)
            generated, conditions = sampler.p_sample_loop(model=model, n=batch_size, c=conditions, a=analyses, resolution=resolution)
            targets = tensor_to_PIL(targets)
            generated = tensor_to_PIL(generated)
            conditions = tensor_to_PIL(conditions)

            targets_list.extend(targets)
            generated_list.extend(generated)
            conditions_list.extend(conditions)

    return targets_list, generated_list, conditions_list

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def mse(imageA, imageB):
    """
    # The 'mean' function calculates the average of the array elements.
    The 'square' function calculates the squared value of each element.
    np.subtract(imageA, imageB) computes the difference between the images.
    """
    err = np.square(np.subtract(imageA, imageB))
    mean_err = np.mean(err)/255**2
    max_err = np.max(err)/255**2
    return mean_err, max_err

def mae_effective_area(imageA, imageB, threshold=254):
    """
    Calculates the Mean Absolute Error (MAE) between two images,
    considering only the effective area (excluding white parts).

    Args:
        imageA: First image (numpy array).
        imageB: Second image (numpy array).
        threshold: Threshold to determine white pixels (default: 250).

    Returns:
        MAE value on the effective area.
    """

    # Create a mask to identify the effective area (non-white pixels)
    # mask = (imageA[:, :, 0] < threshold) & (imageA[:, :, 1] < threshold) & (imageA[:, :, 2] < threshold) | \
    #        (imageB[:, :, 0] < threshold) & (imageB[:, :, 1] < threshold) & (imageB[:, :, 2] < threshold)

    # Apply the mask to both images (for MAE calculation)
    effective_imageA = imageA
    effective_imageB = imageB

    return mae(effective_imageA, effective_imageB)

def mae(imageA, imageB):
    mae = np.mean(np.abs(np.subtract(imageA.astype(np.float32), imageB.astype(np.float32))))/255
    return mae

def calculate_metrics(image_set1: list[Image.Image], image_set2: list[Image.Image]):

    if len(image_set1) != len(image_set2):
        raise ValueError("Number of images in image sets do not match")

    ssim_values = []
    psnr_values = []
    mse_mean_values = []
    mse_max_values = []
    mae_values = []

    for i in range(len(image_set1)):
        ssim_values.append(ssim(np.array(image_set1[i]), np.array(image_set2[i]), win_size=3))
        psnr_values.append(psnr(np.array(image_set1[i]), np.array(image_set2[i]), data_range=255))
        mae_values.append(mae_effective_area(np.array(image_set1[i]), np.array(image_set2[i])))
        mse_mean, mse_max = mse(np.array(image_set1[i]), np.array(image_set2[i]))
        mse_mean_values.append(mse_mean)
        mse_max_values.append(mse_max)

    return ssim_values, psnr_values, mse_mean_values, mse_max_values, mae_values

def sample_save_metrics(
        model: torch.nn.Module,
        device: torch.device,
        sampler: DDPM_Tools,
        image_dataset_path = None,
        structure_dataset_path = None,
        test_path: str = None,
        output_path: str = None,
        length: int = 10,
        test_dataloader: DataLoader = None,
        **kwargs
):

    batch_size = kwargs.get("batch_size")
    resolution = kwargs.get("resolution")
    dataloader = test_dataloader

    sample_path = os.path.join(output_path, "samples")
    target_path = os.path.join(output_path, "targets")
    condition_path = os.path.join(output_path, "conditions")

    parameter_count = count_parameters(model)
    targets, samples, conditions = sample_model_output(model=model, device=device, sampler=sampler, dataloader=dataloader, length=length, batch_size=batch_size, resolution=resolution)
    ssim_values, psnr_values, mse_mean_values, mse_max_values, mae_values = calculate_metrics(targets, samples)
    print(f"SSIM: {(np.mean(ssim_values)):.2f}, PSNR: {(np.mean(psnr_values)):.2f}, MAE: {(np.mean(mae_values)):.2e}, MSE Mean: {(np.mean(mse_mean_values)):.2e}, MSE Max: {(np.mean(mse_max_values)):.2e}, Parameters: {parameter_count}")

    save_image_list(targets, target_path)
    save_image_list(samples, sample_path)
    save_image_list(conditions, condition_path)

def calculate_error_image(target: Image, sample: Image):
    target_array = np.array(target)
    sample_array = np.array(sample)

    error_array = np.abs(target_array-sample_array)

    error_image = Image.fromarray(error_array)

    return error_image

def error_image(structure: Image, target: Image, sample: Image):
    """
    Generates an error image highlighting differences between target and sample images,
    overlayed onto the structure image. Uses a continuous red-to-green color gradient
    for error magnitudes.

    Args:
        structure: Image representing the underlying structure.
        target: Target image for comparison.
        sample: Sample image to compare against the target.

    Returns:
        Image with the error visualization.
    """

    sample_data = sample.getdata()
    target_data = target.getdata()
    structure_data = structure.getdata()
    mask_data = []

    for pixel1, pixel2, pixel3 in zip(sample_data, target_data, structure_data):
        diffs = [abs(pixel1[i] - pixel2[i]) for i in range(len(pixel1))]
        mae = sum(diffs) / (len(pixel1) * 255) * 100

        # Calculate color based on continuous gradient
        if mae > 6:  # Cap at 5% for red
            mae = 6
        hue = 120 - mae * 20  # 120 is green, 0 is red
        color_hex = ImageColor.getrgb(f"hsl({hue}, 100%, 50%)")  # Full saturation, 50% lightness

        if mae > 0.1:  # Only apply color to errors above 1%
            mask_data.append(color_hex)
        else:
            mask_data.append(pixel3)  # Use structure color for low errors

    mask = Image.new(sample.mode, sample.size)
    mask.putdata(mask_data)
    return mask

def comparison_plot(conditions: list, targets: list, samples: list, path: str = None, cbar_width=0.02):
    """
    Creates a comparison plot with target, sample, and error images, including a color bar for the error visualization.
    """
    fig, axs = plt.subplots(len(conditions), 3, figsize=(9, 9))

    # Set column titles
    axs[0, 0].set_title('Exact')
    axs[0, 1].set_title('Prediction')
    axs[0, 2].set_title('Error')

    # Create the colormap and normalization for the color bar
    mae_values = np.linspace(0, 6, 256)  # MAE range from 1% to 5%
    colors = []
    for mae in mae_values:
        hue = 120 - (mae) * 20  # 120 is green, 0 is red
        color_rgb = ImageColor.getrgb(f"hsl({hue}, 100%, 50%)")
        color_normalized = tuple(c / 255 for c in color_rgb)
        colors.append(color_normalized)

    cmap = mcolors.LinearSegmentedColormap.from_list('error_cmap', colors)
    norm = mcolors.Normalize(vmin=0, vmax=6)  # Normalize to 1%-5% range

    for i in range(len(conditions)):
        # Get the image dimensions
        height, width = targets[i].size

        # Plot the target image
        axs[i, 0].imshow(targets[i], extent=[0, width, height, 0])

        # Plot the sample image
        axs[i, 1].imshow(samples[i], extent=[0, width, height, 0])

        # Calculate and plot the error image
        error_img = error_image(conditions[i], targets[i], samples[i])
        im = axs[i, 2].imshow(error_img, extent=[0, width, height, 0])

    # Add the color bar at the end (after all error images are plotted)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs, orientation='vertical',
                        fraction=cbar_width, pad=0.05, label='Absolute Error (%)')

    if path is not None:
        plt.savefig(path)

    plt.show()

def forward_process_image(sampler, dataloader, device):
    if sampler.conditioned_prior == True:
        sampler.init_prior_mean_variance(dataloader)
    nr_images = 5
    images, _, _ = next(iter(dataloader))
    images = images.to(device)
    images_list = []
    noise_steps = sampler.noise_steps
    images_list.append(tensor_to_PIL(images)[0])

    for t in range(0, noise_steps, noise_steps // nr_images):
        t = sampler.get_specific_timesteps(t, images.shape[0])
        noised_images, _ = sampler.noise_images(images, t)
        if sampler.conditioned_prior == True:
            mean = sampler.prior_to_batchsize(sampler.prior_mean, images.shape[0])
            noised_images = noised_images
        noised_images = tensor_to_PIL(noised_images)
        images_list.append(noised_images[0])

        # Calculate the figure size
        fig_width = nr_images * images_list[0].size[0] / 100  # Convert pixels to inches
        fig_height = images_list[0].size[1] / 100  # Convert pixels to inches

    fig, axs = plt.subplots(1, len(images_list))

    for i, img in enumerate(images_list):
        axs[i].imshow(img)
        axs[i].axis('off')  # Hide axes for better visualization

    plt.show()