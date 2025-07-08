import re
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from scipy.interpolate import interp1d


def extract_label(filename: str):
    """
    Extracts the label from the filename based on a specific pattern.
    Args:
        filename: The name of the file from which to extract the label.
    Returns:
        A tuple containing the extracted values: (F, dx, I, angle, Tmax, Tmin).
    """
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

def smooth_interpolation(pixels: tuple, target_length: int) -> list:
    """
    Smoothly interpolate a list of pixel values to a target length using cubic interpolation,
    and return grayscale pixel values (0-255).

    Args:
        pixels: List of pixel values to be interpolated.
        target_length: The desired length of the output list.

    Returns:
        A list of grayscale pixel values (int, 0-255) interpolated to the target length.
    """
    extended_pixels = list(pixels) + [pixels[0]]
    x = np.linspace(0, len(extended_pixels) - 1, num=len(extended_pixels))
    f = interp1d(x, extended_pixels, kind='cubic')
    x_new = np.linspace(0, len(extended_pixels) - 1, num=target_length)
    gray_pixels = [int(np.clip(f(i), 0, 255)) for i in x_new]
    return gray_pixels


def value2graypixel(value: float) -> tuple:
    """
    Convert a normalized value (0 to 1) to a grayscale pixel.

    Args:
        value: A float value between 0 and 1.
        
    Returns:
        A tuple representing the pixel in grayscale.
    """
    gray = int(np.clip(value, 0, 1) * 255)
    return gray

def s2c_angle(image: Image, list: list, Tj: int, image_size: int) -> Image:
    """
    Create a circular image with a gradient based on the pixels in the list.
    Each angle in the list corresponds to a color in the image.

    Args:
        image: the original image to draw on
        list: the pixel values for the edges of the circuit
        Tj: the base pixel value for the center of the image
        image_size: the size of the image to be created
    """
    if image is None:
        image = Image.new("L", (image_size, image_size), 0)
    center = image_size // 2
    radius = image_size // 2
    for y in range(image_size):
        for x in range(image_size):
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance < radius:
                l2s_angle = np.arctan2(dy, dx)
                angle_index = int(((l2s_angle + np.pi) / (2 * np.pi)) * len(list)) % len(list)
                edge_value = list[angle_index]
                radial_ratio = distance / radius
                pixel_value = int(Tj + (edge_value - Tj) * radial_ratio)
                image.putpixel((x, y), pixel_value)
            else:
                image.putpixel((x, y), 0)
    return image

def s2r_angle(image: Image, list: list, Tj: int, image_size: int, r_i, r_o, r_c) -> Image:
    """
    Create a circular image with a gradient based on the pixels in the list.
    Each angle in the list corresponds to a color in the image.

    Args:
        image: the original image to draw on
        list: the pixel values for the edges of the circuit
        Tj: the base pixel value for the center of the image
        image_size: the size of the image to be created
    """
    if image is None:
        image = Image.new("L", (image_size, image_size), 0)
    center = image_size // 2
    radius = image_size // 2
    radius_i = int(r_i / r_c * radius)
    radius_o = int(r_o / r_c * radius)
    for y in range(image_size):
        for x in range(image_size):
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if radius_i < distance < radius_o:
                l2s_angle = np.arctan2(dy, dx)
                angle_index = int(((l2s_angle + np.pi) / (2 * np.pi)) * len(list)) % len(list)
                edge_value = list[angle_index]
                radial_ratio = distance / radius
                pixel_value = int(Tj + (edge_value - Tj) * radial_ratio)
                image.putpixel((x, y), pixel_value)
    return image

def s2c_distance(image: Image, list: list, image_size: int) -> Image:
    """
    Create a circular image with a gradient based on the pixels in the list.
    Each distance in the list corresponds to a color in the image.

    Args:
        image: the original image to draw on
        list: the pixel values for the edges of the circuit
        image_size: the size of the image to be created
    """
    if image is None:
        image = Image.new("L", (image_size, image_size), 0)
    center = image_size // 2
    radius = image_size // 2
    for y in range(image_size):
        for x in range(image_size):
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance < radius:
                distance_index = int((distance / radius) * len(list)) % len(list)
                image.putpixel((x, y), list[distance_index])
    return image

def s2r_distance(image, list, image_size, r_i, r_o, r_c) -> Image:
    """
    Create a ring image with a gradient based on the pixels in the list.
    Each distance in the list corresponds to a color in the image.

    Args:
        image: the original image to draw on
        list: the pixel values for the radius of the ring
        image_size: the size of the image to be created
        r_i: the inner radius of the ring
        r_o: the outer radius of the ring
        r_c: the outer radius of the circular
    """
    if image is None:
        image = Image.new("L", (image_size, image_size), 0)
    center = image_size // 2
    radius = image_size // 2
    radius_i = int(r_i / r_c * radius)
    radius_o = int(r_o / r_c * radius)
    for y in range(image_size):
        for x in range(image_size):
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if radius_i < distance < radius_o:
                distance_index = int((distance / radius) * len(list)) % len(list)
                image.putpixel((x, y), list[distance_index])
    return image

def image_add_mask(image: Image, margin: int) -> Image:
    """
    Add a circular mask to the image to create a circular effect.

    Args:
        image: The original image to add the mask.
        margin: The margin to apply to the mask.
    """
    image_size = image.size[0]
    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((margin, margin, image_size - margin, image_size - margin), fill=255)
    image.putalpha(mask)
    return image

def image_add_0(image: Image) -> Image:
    """
    Remove pixels outside a circular area from the image.
    Args:
        image: The original image to process.
    """
    image_size = image.size[0]
    center = image_size // 2
    radius = image_size // 2
    for y in range(image_size):
        for x in range(image_size):
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance > radius - 2: # Allow a small margin
                image.putpixel((x, y), 0)
    return image

def read_t_range(range_file: str, F: int, dx: int, I: int) -> tuple:
    """
    Read the t_range file and filter the data based on force, misplacement, and current.

    Args:
        range_file: Path to the t_range file.
        F: The force value (kN) to filter by.
        dx: The misplacement value (mm) to filter by.
        I: The current value (A) to filter by.
    """
    t_range = pd.read_csv(range_file)
    row = t_range[(t_range["F (kN)"] == F) & (t_range["dx (mm)"] == dx) & (t_range["I (A)"] == I)]
    if not row.empty:
        surface_max = float(row["surface_max (°C)"].values[0])
        surface_min = float(row["surface_min (°C)"].values[0])
        side_max = float(row["side_max (°C)"].values[0])
        side_min = float(row["side_min (°C)"].values[0])
        return surface_max, surface_min, side_max, side_min
    else:
        raise ValueError(f"No matching row for F={F}, dx={dx}, I={I}")
    
def add_t_range(range_file: str, F: int, dx: int, I: int, add_name: str, add_value: float):
    """
    Add a new value to the t_range file for a specific force, misplacement, and current.

    Args:
        range_file: Path to the t_range file.
        F: The force value (kN) to filter by.
        dx: The misplacement value (mm) to filter by.
        I: The current value (A) to filter by.
        add_name: The name of the column to add the value to.
        add_value: The value to add.
    """
    t_range = pd.read_csv(range_file)
    if add_name not in t_range.columns:
        t_range[add_name] = 0.0

    row = t_range[(t_range["F (kN)"] == F) & (t_range["dx (mm)"] == dx) & (t_range["I (A)"] == I)]
    if not row.empty:
        row_index = row.index[0]
        add_value = round(add_value, 2)
        t_range.loc[row_index, add_name] = add_value
        t_range.to_csv(range_file, index=False)
    else:
        raise ValueError(f"No matching row for F={F}, dx={dx}, I={I}, angle={angle}")

def sigmoid(x):
    return 1 / (1 + np.exp(- x))