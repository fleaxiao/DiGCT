import numpy as np

from PIL import Image, ImageDraw
import pandas as pd


def s2c_angle(list: list, image_size: int) -> Image:
    """
    Create a circular image with a gradient based on the pixels in the list.
    Each angle in the list corresponds to a color in the image.

    args:
        list: the pixel values for the edges of the circuit
        image_size: the size of the image to be created
    """
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

def s2c_distance(list: list, image_size: int) -> Image:
    """
    Create a circular image with a gradient based on the pixels in the list.
    Each distance in the list corresponds to a color in the image.

    args:
        list: the pixel values for the edges of the circuit
        image_size: the size of the image to be created
    """
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
    Draw a ring on the image.

    args:
        image: the image to draw on
        center: the center of the circle
        radius: the radius of the circle
        color: the color of the circle
    """

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

def read_t_range(range_file, force, misplacement, current):
    
    t_range = pd.read_csv(range_file)
    filtered_data = t_range[(t_range['F (kN)'] == force) & t_range['dx (mm)'] == misplacement & (t_range['I (A)'] == current)]
    surface_max = float(filtered_data['surface_max (°C)'].values[0]) if not filtered_data.empty else None
    surface_min = float(filtered_data['surface_min (°C)'].values[0]) if not filtered_data.empty else None
    side_max = float(filtered_data['side_max (°C)'].values[0]) if not filtered_data.empty else None
    side_min = float(filtered_data['side_min (°C)'].values[0]) if not filtered_data.empty else None

def sigmoid(x):
    return 1 / (1 + np.exp(- x))