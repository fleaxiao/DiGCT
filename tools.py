import numpy as np

from PIL import Image, ImageDraw
import pandas as pd


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
        return surface_max, surface_min
    else:
        raise ValueError(f"No matching row for F={F}, dx={dx}, I={I}")


def sigmoid(x):
    return 1 / (1 + np.exp(- x))