import numpy as np

from PIL import Image, ImageDraw
import pandas as pd


def s2c_angle(list, image_size):
    """
    Create a circular image with a gradient based on the angles in the list.
    Each angle in the list corresponds to a color in the image.
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

def read_t_range(range_file, force, misplacement, current):
    
    t_range = pd.read_csv(range_file)
    filtered_data = t_range[(t_range['F (kN)'] == force) & t_range['dx (mm)'] == misplacement & (t_range['I (A)'] == current)]
    surface_max = float(filtered_data['surface_max (°C)'].values[0]) if not filtered_data.empty else None
    surface_min = float(filtered_data['surface_min (°C)'].values[0]) if not filtered_data.empty else None
    side_max = float(filtered_data['side_max (°C)'].values[0]) if not filtered_data.empty else None
    side_min = float(filtered_data['side_min (°C)'].values[0]) if not filtered_data.empty else None

    print("Surface max for F = 60 kN, dx = 10 mm, I = 0 A:", surface_max)
