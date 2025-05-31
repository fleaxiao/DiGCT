import os
import re
import yaml
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image, ImageDraw

from data_tool import smooth_interpolation, s2c_angle


def data_preprocess(args):

    # Load config file
    DATA_PATH = args.data_path
    DATASET_PATH = args.dataset_path
    IMAGE_SIZE = args.image_size
    ANGLE_STEP = args.angle_step

    surface_start = 378
    surface_end = 2526
    side_start = 132
    side_end = 2772

    surface_center = surface_start + (surface_end - surface_start) // 2
    surface_radius = (surface_end - surface_start) // 2
    side_length = side_end - side_start

    S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
    L_PATH = os.path.join(DATASET_PATH, "IGCT", "L")
    L2S_PATH = os.path.join(DATASET_PATH, "IGCT", "L2S")
    P2S_PATH = os.path.join(DATASET_PATH, "IGCT", "P2S")

    os.makedirs(DATASET_PATH, exist_ok=True)
    for path in [S_PATH, L_PATH, L2S_PATH, P2S_PATH]:
        os.makedirs(path, exist_ok=True)

    for filename in tqdm(os.listdir(DATA_PATH), desc="Processing files"):

        ## Surface images
        if filename.startswith("surface") and filename.endswith(".png") and not filename.endswith("_0.png"):
            input_path = os.path.join(DATA_PATH, filename)
            
            with Image.open(input_path) as img:
                img = img.convert("RGB")

                width, height = img.size
                left = surface_center - surface_radius
                top = height // 2 - surface_radius
                right = surface_center + surface_radius
                bottom = height // 2 + surface_radius
                
                cropped_img = img.crop((left, top, right, bottom))
                
                mask = Image.new("L", (right - left, bottom - top), 0)
                draw = ImageDraw.Draw(mask)

                draw.ellipse((0, 0, right - left, bottom - top), fill=255)
                cropped_img.putalpha(mask)

                for angle in range(0, 360, ANGLE_STEP):
                    rotated_img = cropped_img.rotate(angle, expand=False)
                    rotated_img = rotated_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

                    name, ext = os.path.splitext(filename)
                    f_number_match = re.search(r"F_(-?\d+)", name)
                    if f_number_match:
                        f_number = int(abs(int(f_number_match.group(1))) * 2 / 1000)
                        name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                    dx_number_match = re.search(r"dx_([\d\.]+)", name)
                    if dx_number_match:
                        dx_number = int(float(dx_number_match.group(1)) * 1000)
                        name = re.sub(r"dx_([\d\.]+)", f"dx_{dx_number}", name)
                    name = name.replace("surface_", "")
                    output_filename = f"{name}_angle_{angle}{ext}"
                    output_path = os.path.join(S_PATH, output_filename)
                    rotated_img.save(output_path)
            
        ## Side images
        if filename.startswith("side") and filename.endswith(".png") and not filename.endswith("_0.png"):
            input_path = os.path.join(DATA_PATH, filename)

            with Image.open(input_path) as img:
                img = img.convert("RGBA")

                width, height = img.size
                center_line = height // 2
                pixels = []
                for x in range(side_start, side_end):
                    pixel = img.getpixel((x, center_line))
                    pixels.append(pixel)

                # Original side image
                pixels_extend = pixels.copy()
                print(pixels_extend)
                reversed_pixels = list(reversed(pixels))
                pixels_extend.extend(reversed_pixels)
                pixels_extend.extend(pixels)

                for angle in range(0, 360, ANGLE_STEP):
                    start_pixel = int(2 * angle * side_length / 360)
                    pixels_rotated = pixels_extend[start_pixel:start_pixel + side_length]

                    img_rotated = Image.new("RGBA", (len(pixels_rotated), 1), (0, 0, 0, 0))
                    img_rotated.putdata(pixels_rotated)
                    img_rotated = img_rotated.resize((IMAGE_SIZE, 1), Image.Resampling.LANCZOS)

                    name, ext = os.path.splitext(filename)
                    f_number_match = re.search(r"F_(-?\d+)", name)
                    if f_number_match:
                        f_number = int(abs(int(f_number_match.group(1))) * 2 / 1000)
                        name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                    dx_number_match = re.search(r"dx_([\d\.]+)", name)
                    if dx_number_match:
                        dx_number = int(float(dx_number_match.group(1)) * 1000)
                        name = re.sub(r"dx_([\d\.]+)", f"dx_{dx_number}", name)
                    name = name.replace("side_", "")
                    output_filename = f"{name}_angle_{angle}{ext}"
                    output_path = os.path.join(L_PATH, output_filename)
                    img_rotated.save(output_path)

                # p2s side image
                pixels_extend = pixels.copy()
                reversed_pixels = list(reversed(pixels))
                pixels_extend.extend(reversed_pixels)
                for angle in range(0, 360, ANGLE_STEP):
                    start_pixel = int(2 * angle * side_length / 360)
                    interval = side_length // 4
                    p2s_list = [pixels_extend[(start_pixel + i * interval) % len(pixels_extend)] for i in range(8)]
                    p2s_list = smooth_interpolation(p2s_list, IMAGE_SIZE + 1)
                    p2s_image = s2c_angle(None, p2s_list, IMAGE_SIZE)

                    name, ext = os.path.splitext(filename)
                    f_number_match = re.search(r"F_(-?\d+)", name)
                    if f_number_match:
                        f_number = int(abs(int(f_number_match.group(1))) * 2 / 1000)
                        name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                    dx_number_match = re.search(r"dx_([\d\.]+)", name)
                    if dx_number_match:
                        dx_number = int(float(dx_number_match.group(1)) * 1000)
                        name = re.sub(r"dx_([\d\.]+)", f"dx_{dx_number}", name)
                    name = name.replace("side_", "")
                    output_filename = f"{name}_angle_{angle}{ext}"
                    output_path = os.path.join(P2S_PATH, output_filename)
                    p2s_image.save(output_path)

                # l2s side image
                img_o = Image.new("RGBA", (len(pixels), 1), (0, 0, 0, 0))
                img_o.putdata(pixels)
                img_o = img_o.resize((IMAGE_SIZE, 1), Image.Resampling.LANCZOS)
                l2s_list = list(img_o.getdata())
                l2s_list.extend(reversed(l2s_list))

                l2s_image = s2c_angle(None, l2s_list, IMAGE_SIZE)

                for angle in range(0, 360, ANGLE_STEP):
                    rotated_img = l2s_image.rotate(angle, expand=False)
                    rotated_img = rotated_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

                    name, ext = os.path.splitext(filename)
                    f_number_match = re.search(r"F_(-?\d+)", name)
                    if f_number_match:
                        f_number = int(abs(int(f_number_match.group(1))) * 2 / 1000)
                        name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                    dx_number_match = re.search(r"dx_([\d\.]+)", name)
                    if dx_number_match:
                        dx_number = int(float(dx_number_match.group(1)) * 1000)
                        name = re.sub(r"dx_([\d\.]+)", f"dx_{dx_number}", name)
                    name = name.replace("side_", "")
                    output_filename = f"{name}_angle_{angle}{ext}"
                    output_path = os.path.join(L2S_PATH, output_filename)
                    rotated_img.save(output_path)

    ## Range csv
    t_range = pd.read_csv(os.path.join(DATA_PATH, "T_range.csv"), header=4)
    t_range.rename(columns={t_range.columns[0]: "F (kN)"}, inplace=True)
    t_range["F (kN)"] = t_range["F (kN)"].abs() * 2
    t_range = t_range[t_range["I (A)"] != 0]  # Remove rows where current is zero
    for col in t_range.columns[1:]:
        t_range[col] = t_range[col].round(3)
    t_range.to_csv(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Dataset Preprocessing')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config_preprocess.yml')

    # Load config file
    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    # Data Preprocessing
    data_preprocess(args=args)

    print("Dataset preprocessing is completed.")