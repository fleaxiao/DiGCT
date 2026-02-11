import numpy as np
import pandas as pd
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image, ImageColor

from model_utils import *
from torch.utils.data import DataLoader
from model_diff import VDM_Tools, DDPM_Tools


def sample_model(
        model: torch.nn.Module,
        result_path: str,
        device: torch.device,
        sampler: DDPM_Tools,
        dataloader: DataLoader,
        resolution: int,
        length: int,
        batch_size: int,
        physics_constraint: bool = True
):
    targets_value_list = []
    targets_pixel_list = []
    targets_max = []
    targets_min = []

    generations_value_list = []
    generations_pixel_list = []
    generations_max = []
    generations_min = []

    conditions_value_list = []
    conditions_pixel_list = []
    conditions_max = []
    conditions_min = []

    F_list = []
    theta_list = []
    vf_list = []
    I_list = []
    angle_list = []

    iterator = iter(dataloader)

    dataset_range = pd.read_csv(os.path.join(result_path, "dataset_range.csv"))
    surface_max = dataset_range["surface_max"].values[0]
    surface_min = dataset_range["surface_min"].values[0]
    side_max = dataset_range["side_max"].values[0]
    side_min = dataset_range["side_min"].values[0]
    condition_max = dataset_range["condition_max"].values[0]
    condition_min = dataset_range["condition_min"].values[0]
    gap_max = dataset_range["gap_max"].values[0]
    gap_min = dataset_range["gap_min"].values[0]

    for i in range(0, length * batch_size, batch_size):
        targets, conditions, F, theta, vf, I, angle = next(iterator)
        conditions, targets = conditions.to(device), targets.to(device)
        generations, conditions = sampler.p_sample_loop(model=model, n=batch_size, c=conditions,  resolution=resolution)

        if physics_constraint == True:
            g = ((generations + 1) / 2) * (gap_max - gap_min) + gap_min
            t = ((targets + 1) / 2) * (gap_max - gap_min) + gap_min
            c = ((conditions + 1) / 2) * (condition_max - condition_min) + condition_min

            g = g + c
            t = t + c

            generations = g
            generations = (generations - surface_min) / (surface_max - surface_min)
            generations = ((generations * 2) - 1)
            targets = t
            targets = (targets - surface_min) / (surface_max - surface_min)
            targets = ((targets * 2) - 1)

        targets_value, targets_pixel, targets_max_batch, targets_min_batch = tensor_to_PIL_range(targets, surface_max, surface_min)
        generations_value, generations_pixel, generations_max_batch, generations_min_batch = tensor_to_PIL_range(generations, surface_max, surface_min)
        conditions_value, conditions_pixel, conditions_max_batch, conditions_min_batch = tensor_to_PIL_range(conditions, condition_max, condition_min)

        targets_value_list.extend(targets_value)
        generations_value_list.extend(generations_value)
        conditions_value_list.extend(conditions_value)

        targets_pixel_list.extend(targets_pixel)
        targets_max.extend(targets_max_batch)
        targets_min.extend(targets_min_batch)

        generations_pixel_list.extend(generations_pixel)
        generations_max.extend(generations_max_batch)
        generations_min.extend(generations_min_batch)

        conditions_pixel_list.extend(conditions_pixel)
        conditions_max.extend(conditions_max_batch)
        conditions_min.extend(conditions_min_batch)

        F_list.extend(F)
        theta_list.extend(theta)
        vf_list.extend(vf)
        I_list.extend(I)
        angle_list.extend(angle)

    return targets_value_list, generations_value_list, conditions_value_list, targets_pixel_list, generations_pixel_list, conditions_pixel_list, targets_max, targets_min, generations_max, generations_min, conditions_max, conditions_min, F_list, theta_list, vf_list, I_list, angle_list

def calculate_metrics(image_set1: list[Image.Image], image_set2: list[Image.Image]):

    if len(image_set1) != len(image_set2):
        raise ValueError("Number of images in image sets do not match")

    ssim_values = []
    psnr_values = []
    mse_mean_values = []
    mse_max_values = []
    mae_values = []
    max_err_values = []
    var_err_values = []

    for i in range(len(image_set1)):
        ssim_values.append(ssim(np.array(image_set1[i]), np.array(image_set2[i]), win_size=3))
        psnr_values.append(psnr(np.array(image_set1[i]), np.array(image_set2[i]), data_range=255))
        mse_mean, mse_max = mse(np.array(image_set1[i]), np.array(image_set2[i]))
        mse_mean_values.append(mse_mean)
        mse_max_values.append(mse_max)
        mae_values.append(mae(np.array(image_set1[i]), np.array(image_set2[i])))
        max_err_values.append(max_err(np.array(image_set1[i]), np.array(image_set2[i])))
        var_err_value = var_err(np.array(image_set1[i]), np.array(image_set2[i]))

    return ssim_values, psnr_values, mse_mean_values, mse_max_values, mae_values, max_err_values, var_err_value

def show_metrics(
        result_path: str,
        test_dataloader: DataLoader = None,
        condition_images: list[Image.Image] = None,
        target_images: list[Image.Image] = None,
        generation_images: list[Image.Image] = None,
        **kwargs
):

    dataset_range = pd.read_csv(os.path.join(result_path, "dataset_range.csv"))
    surface_max = dataset_range["surface_max"].values[0]
    surface_min = dataset_range["surface_min"].values[0]

    ssim_values, psnr_values, mse_mean_values, mse_max_values, mae_values, max_err_values, var_err_values = calculate_metrics(target_images, generation_images)

    mae_temperatures = np.array(mae_values) * (surface_max - surface_min)
    max_err_temperatures = np.array(max_err_values) * (surface_max - surface_min)
    var_err_temperatures = np.array(var_err_values) * (surface_max - surface_min)

    print(f"SSIM: {(np.mean(ssim_values)):.3f}, PSNR: {(np.mean(psnr_values)):.3f}, MAE: {np.mean(mae_values):.3f}, MAX: {np.max(max_err_values):.3f}, VAR: {np.mean(var_err_values):.3f}")
    print(f"Mean temperature error: {np.mean(mae_temperatures):.3f}, Max temperature error: {np.mean(max_err_temperatures):.3f}")

def sample_show_metrics(
        model: torch.nn.Module,
        result_path: str,
        device: torch.device,
        sampler: DDPM_Tools,
        sample_path: str = None,
        length: int = 10,
        test_dataloader: DataLoader = None,
        **kwargs
):

    batch_size = kwargs.get("batch_size")
    resolution = kwargs.get("resolution")
    physics_constraint = kwargs.get("physics_constraint")
    dataloader = test_dataloader

    target_path = os.path.join(sample_path, "target")
    generation_path = os.path.join(sample_path, "generation")
    condition_path = os.path.join(sample_path, "condition")

    parameter_count = count_parameters(model)
    print(f"\nParameter Count: {parameter_count}")

    dataset_range = pd.read_csv(os.path.join(result_path, "dataset_range.csv"))
    surface_max = dataset_range["surface_max"].values[0]
    surface_min = dataset_range["surface_min"].values[0]

    targets_value, generations_value, conditions_value, targets_pixel, generations_pixel, conditions_pixel, targets_max, targets_min, generations_max, generations_min, conditions_max, conditions_min, F, theta, vf, I, angle = sample_model(model=model, result_path=result_path, device=device, sampler=sampler, dataloader=dataloader, length=length, batch_size=batch_size, resolution=resolution, physics_constraint=physics_constraint)

    ssim_values, psnr_values, _, _, mae_values, max_err_values, var_err_values = calculate_metrics(targets_value, conditions_value)
    mae_temperatures = np.array(mae_values) * (surface_max - surface_min)
    max_err_temperatures = np.array(max_err_values) * (surface_max - surface_min)
    var_err_temperatures = np.array(var_err_values) * (surface_max - surface_min)

    print(f"SSIM: {(np.mean(ssim_values)):.3f}, PSNR: {(np.mean(psnr_values)):.3f}, MAE: {np.mean(mae_values):.3f}, MAX: {np.max(max_err_values):.3f}, VAR: {np.mean(var_err_values):.3f}")
    print(f"Mean temperature error: {np.mean(mae_temperatures):.3f}, Max temperature error: {np.mean(max_err_temperatures):.3f}")

    save_image_list(targets_pixel, F, theta, vf,  I, angle, targets_max, targets_min, target_path)
    save_image_list(generations_pixel, F, theta, vf, I, angle, generations_max, generations_min, generation_path)
    save_image_list(conditions_pixel, F, theta, vf, I, angle, conditions_max, conditions_min, condition_path)

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