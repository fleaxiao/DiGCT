import numpy as np
import pandas as pd

from PIL import Image
from scipy.interpolate import interp1d


def smooth_interpolation(pixels: tuple, target_length: int) -> list:
    """
    Smoothly interpolate a list of pixel values to a target length using cubic interpolation.

    Args:
        pixels: List of pixel values to be interpolated.
        target_length: The desired length of the output list.
    
    Returns:
        A list of pixel values interpolated to the target length.
    """
    extended_pixels = pixels + [pixels[0]]
    x = np.linspace(0, len(extended_pixels) - 1, num=len(extended_pixels))
    f = interp1d(x, extended_pixels, kind='cubic', axis=0) 
    x_new = np.linspace(0, len(extended_pixels) - 1, num=target_length) 
    return [tuple(map(int, f(i))) for i in x_new]

def value2graypixel(value: float) -> tuple:
    """
    Convert a normalized value (0 to 1) to a grayscale RGB tuple.

    Args:
        value: A float value between 0 and 1.
        
    Returns:
        A tuple representing the RGB color in grayscale.
    """
    gray = int(np.clip(value, 0, 1) * 255)
    return (gray, gray, gray)

def s2c_angle(image: Image, list: list, image_size: int) -> Image:
    """
    Create a circular image with a gradient based on the pixels in the list.
    Each angle in the list corresponds to a color in the image.

    Args:
        image: the original image to draw on
        list: the pixel values for the edges of the circuit
        image_size: the size of the image to be created
    """
    if image is None:
        image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
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
                image.putpixel((x, y), list[angle_index])
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
        image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
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
        image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
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
        t_range.loc[row_index, add_name] = round(add_value, 2)
        t_range.to_csv(range_file, index=False)
    else:
        raise ValueError(f"No matching row for F={F}, dx={dx}, I={I}")

def sigmoid(x):
    return 1 / (1 + np.exp(- x))