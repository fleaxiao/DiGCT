import os
import re
import yaml
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

from data_tool import *
from data_ana import analytical_model


def data_preprocess(args):
    # Load config file
    DATA_PATH = args.data_path
    DATASET_PATH = args.dataset_path
    IMAGE_SIZE = args.image_size
    ANGLE_STEP = args.angle_step
    MARGIN = args.margin
    os.makedirs(os.path.join(DATASET_PATH, "IGCT"), exist_ok=True)

    SURFACE = args.surface
    SIDE = args.side
    L2S = args.L2S
    P2S = args.P2S
    PA2S = args.PA2S
    G = args.G

    R_node_1 = float(args.R_node_1)
    R_node_2 = float(args.R_node_2)
    R_node_3 = float(args.R_node_3)
    L_node = float(args.L_node)
    R_chip = float(args.R_chip)
    L_chip = float(args.L_chip)

    surface_start = 378
    surface_end = 2526
    side_start = 132
    side_end = 2772

    surface_center = surface_start + (surface_end - surface_start) // 2
    surface_radius = (surface_end - surface_start) // 2
    side_length = side_end - side_start

    # Process csv file
    t_range = pd.read_csv(os.path.join(DATA_PATH, "T_range.csv"), header=4)
    t_range.rename(columns={t_range.columns[0]: "F (kN)"}, inplace=True)
    t_range["F (kN)"] = t_range["F (kN)"].abs() * 2
    t_range = t_range[t_range["I (A)"] != 0]
    for col in t_range.columns[1:]:
        t_range[col] = t_range[col].round(2)

    # angle_rows = []
    # for _, row in t_range.iterrows():
    #     for angle in range(0, 360, ANGLE_STEP):
    #         new_row = row.copy()
    #         cols = list(new_row.index)
    #         cols.insert(cols.index('I (A)') + 1, 'angle (°)')
    #         new_row = new_row.reindex(cols)
    #         new_row['angle (°)'] = angle
    #         angle_rows.append(new_row)
    # t_range = pd.DataFrame(angle_rows)
    t_range.to_csv(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), index=False)

    # Process image file
    surface_files = [f for f in os.listdir(DATA_PATH) if f.startswith("surface") and f.endswith(".png") and not f.endswith("_0.png")]
    side_files = [f for f in os.listdir(DATA_PATH) if f.startswith("side") and f.endswith(".png") and not f.endswith("_0.png")]

    ## Process surface images
    if SURFACE:
        S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
        os.makedirs(S_PATH, exist_ok=True)
        for filename in tqdm(surface_files, desc="Processing surface images"):
            if filename.startswith("surface") and filename.endswith(".png") and not filename.endswith("_0.png"):
                parts = filename.split('_')
                F = int(abs(float(parts[2])) * 2 / 1000)
                dx = int(float(parts[4]) * 1000)
                I = int(os.path.splitext(parts[6])[0])
                surface_max, surface_min, _, _ = read_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, dx, I)

                with Image.open(os.path.join(DATA_PATH, filename)).convert("L") as img:
                    width, height = img.size
                    left = surface_center - surface_radius
                    top = height // 2 - surface_radius
                    right = surface_center + surface_radius
                    bottom = height // 2 + surface_radius

                    surface_img = img.crop((left, top, right, bottom))
                    surface_img = surface_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

                    for angle in range(0, 360, ANGLE_STEP):
                        rotated_img = surface_img.rotate(angle, expand=False)
                        rotated_img = image_add_0(rotated_img, margin=MARGIN) # critical to deal with the COMOSL generated image
                        rotated_img = image_add_mask(rotated_img, margin=MARGIN)

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
                        output_filename = f"{name}_angle_{angle}_Tmax_{surface_max}_Tmin_{surface_min}{ext}"
                        output_path = os.path.join(S_PATH, output_filename)
                        rotated_img.save(output_path)
    
    ## Process side images
    if SIDE:
        for filename in tqdm(side_files, desc="Processing side images"):
            parts = filename.split('_')
            F = int(abs(float(parts[2])) * 2 / 1000)
            dx = int(float(parts[4]) * 1000)
            I = int(os.path.splitext(parts[6])[0])
            _, _, side_max, side_min = read_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, dx, I)

            with Image.open(os.path.join(DATA_PATH, filename)).convert("L") as img:
                width, height = img.size
                center_line = height // 2
                pixels = []
                for x in range(side_start, side_end):
                    pixel = img.getpixel((x, center_line))
                    pixels.append(pixel)

                ### original side image
                L_PATH = os.path.join(DATASET_PATH, "IGCT", "L")
                os.makedirs(L_PATH, exist_ok=True)
                pixels_extend = pixels.copy()

                reversed_pixels = list(reversed(pixels))
                pixels_extend.extend(reversed_pixels)
                pixels_extend.extend(pixels)

                for angle in range(0, 360, ANGLE_STEP):
                    start_pixel = int(2 * angle * side_length / 360)
                    pixels_rotated = pixels_extend[start_pixel:start_pixel + side_length]

                    img_rotated = Image.new("L", (len(pixels_rotated), 1), 0)
                    img_rotated.putdata(pixels_rotated)
                    img_rotated = img_rotated.resize((IMAGE_SIZE, 1), Image.Resampling.NEAREST)

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
                    output_filename = f"{name}_angle_{angle}_Tmax_{side_max}_Tmin_{side_min}{ext}"
                    output_path = os.path.join(L_PATH, output_filename)
                    img_rotated.save(output_path)

                ### l2s side image w/o analysis model
                if L2S:
                    L2S_PATH = os.path.join(DATASET_PATH, "IGCT", "L2S")
                    os.makedirs(L2S_PATH, exist_ok=True)

                    img_o = Image.new("L", (len(pixels), 1), 0)
                    img_o.putdata(pixels)
                    img_o = img_o.resize((IMAGE_SIZE, 1), Image.Resampling.NEAREST)
                    l2s_list = list(img_o.getdata())
                    l2s_list.extend(reversed(l2s_list))

                    l2s_image = s2c_angle(None, l2s_list, 255, IMAGE_SIZE)
                    l2s_image = l2s_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

                    for angle in range(0, 360, ANGLE_STEP):
                        rotated_img = l2s_image.rotate(angle, expand=False)
                        rotated_img = image_add_0(rotated_img, margin=MARGIN)
                        rotated_img = image_add_mask(rotated_img, margin=MARGIN)

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
                        output_filename = f"{name}_angle_{angle}_Tmax_{side_max}_Tmin_{side_min}{ext}"
                        output_path = os.path.join(L2S_PATH, output_filename)
                        rotated_img.save(output_path)

                ### p2s side image w/o analysis model
                if P2S:
                    P2S_PATH = os.path.join(DATASET_PATH, "IGCT", "P2S")
                    os.makedirs(P2S_PATH, exist_ok=True)

                    pixels_extend = pixels.copy()
                    reversed_pixels = list(reversed(pixels))
                    pixels_extend.extend(reversed_pixels)

                    for angle in range(0, 360, ANGLE_STEP):
                        start_pixel = int(2 * angle * side_length / 360)
                        interval = side_length // 4
                        p2s_list = np.array([pixels_extend[(start_pixel + i * interval) % len(pixels_extend)] for i in range(8)]) 
                        p2s_list = smooth_interpolation(p2s_list, IMAGE_SIZE + 1)
                        p2s_image = s2c_angle(None, p2s_list, Tj_pixel, IMAGE_SIZE)
                        p2s_image = image_add_0(p2s_image, margin=MARGIN)
                        p2s_image = image_add_mask(p2s_image, margin=MARGIN)

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
                        output_filename = f"{name}_angle_{angle}_Tmax_{side_max}_Tmin_{side_min}{ext}"
                        output_path = os.path.join(P2S_PATH, output_filename)
                        p2s_image.save(output_path)

                ### pa2s side image w analysis model
                if PA2S:
                    PA2S_PATH = os.path.join(DATASET_PATH, "IGCT", "PA2S")
                    os.makedirs(PA2S_PATH, exist_ok=True)

                    Tj, _, _, _ = analytical_model(args, F, dx, I)
                    Ta_max = max(Tj, side_max)
                    Ta_min = min(Tj, side_min)
                    Tj_pixel = int((Tj - Ta_min) / (Ta_max - Ta_min) * 255.0)

                    pixels_extend = pixels.copy()
                    reversed_pixels = list(reversed(pixels))
                    pixels_extend.extend(reversed_pixels)
                    T_extend = np.array(pixels_extend) * (side_max - side_min) / 255.0 + side_min

                    for angle in range(0, 360, ANGLE_STEP):
                        start_pixel = int(2 * angle * side_length / 360)
                        interval = side_length // 4
                        p2s_list = np.array([T_extend[(start_pixel + i * interval) % len(T_extend)] for i in range(8)]) 
                        p2s_list = (p2s_list - Ta_min) / (Ta_max - Ta_min) * 255.0
                        p2s_list = smooth_interpolation(p2s_list, IMAGE_SIZE + 1)
                        p2s_image = s2c_angle(None, p2s_list, Tj_pixel, IMAGE_SIZE)
                        p2s_image = s2r_angle(p2s_image, p2s_list, Tj_pixel * (L_chip - L_node) / L_chip, IMAGE_SIZE, 0, R_node_1, R_chip)
                        p2s_image = s2r_angle(p2s_image, p2s_list, Tj_pixel * (L_chip - L_node) / L_chip, IMAGE_SIZE, R_node_2, R_node_3, R_chip)
                        p2s_image = image_add_0(p2s_image, margin=MARGIN)
                        p2s_image = image_add_mask(p2s_image, margin=MARGIN)

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
                        output_filename = f"{name}_angle_{angle}_Tmax_{Ta_max}_Tmin_{Ta_min}{ext}"
                        output_path = os.path.join(PA2S_PATH, output_filename)
                        p2s_image.save(output_path)

                        add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, dx, I, 'analysis_max (°C)', Ta_max)
                        add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, dx, I, 'analysis_min (°C)', Ta_min)

    ## Process gap images
    if G:
        S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
        if not os.path.exists(S_PATH):
            raise FileNotFoundError(f"Surface images directory {S_PATH} does not exist. Please process surface images first.")
        if not os.listdir(S_PATH):
            raise ValueError(f"Surface images directory {S_PATH} is empty. Please process surface images first.")
        PA2S_PATH = os.path.join(DATASET_PATH, "IGCT", "PA2S")
        G_PATH = os.path.join(DATASET_PATH, "IGCT", "G")
        os.makedirs(G_PATH, exist_ok=True)

        s_files = sorted([f for f in os.listdir(S_PATH) if f.endswith('.png')])
        pa2s_files = sorted([f for f in os.listdir(PA2S_PATH) if f.endswith('.png')])

        for s_file, pa2s_file in tqdm(zip(s_files, pa2s_files), total=min(len(s_files), len(pa2s_files)), desc="Processing gap images"):
            s_name, ext = os.path.splitext(s_file)
            s_img_path = os.path.join(S_PATH, s_file)
            pa2s_img_path = os.path.join(PA2S_PATH, pa2s_file)
            with Image.open(s_img_path).convert("L") as s_img, Image.open(pa2s_img_path).convert("L") as pa2s_img:
                s_F, s_dx, s_I, s_angle, s_Tmax, s_Tmin = extract_label(s_img_path)
                pa2s_F, pa2s_dx, pa2s_I, pa2s_angle, pa2s_Tmax, pa2s_Tmin = extract_label(pa2s_img_path)
                if not (s_F == pa2s_F) or not (s_dx == pa2s_dx) or not (s_I == pa2s_I) or not (s_angle == pa2s_angle):
                    raise ValueError(
                        f"Parameters must match for gap image:\n"
                        f"  F values - Surface: {s_F}, PA2S: {pa2s_F}\n"
                        f"  dx values - Surface: {s_dx}, PA2S: {pa2s_dx}\n"
                        f"  I values - Surface: {s_I}, PA2S: {pa2s_I}\n"
                        f"  angle values - Surface: {s_angle}, PA2S: {pa2s_angle}"
                    )

                s_np = np.array(s_img)
                s_np = s_np.astype(np.float32) / 255.0 * (s_Tmax - s_Tmin) + s_Tmin

                pa2s_np = np.array(pa2s_img)
                pa2s_np = pa2s_np.astype(np.float32) / 255.0 * (pa2s_Tmax - pa2s_Tmin) + pa2s_Tmin

                g_np = s_np - pa2s_np
                Tg_max = round(float(np.max(g_np)), 2)
                Tg_min = round(float(np.min(g_np)), 2)
                g_np = (g_np - Tg_min) / (Tg_max - Tg_min) * 255.0
                g_image = Image.fromarray(g_np.astype(np.uint8), mode='L')
                g_image = image_add_mask(g_image, margin=MARGIN)

                Tg_max_match = re.search(r"Tmax_([\d\.]+)", s_name)
                if  Tg_max_match:
                    g_name = re.sub(r"Tmax_([\d\.]+)", f"Tmax_{Tg_max:.2f}", s_name)
                Tg_min_match = re.search(r"Tmin_([\d\.]+)", s_name)
                if  Tg_min_match:
                    g_name = re.sub(r"Tmin_([\d\.]+)", f"Tmin_{Tg_min:.2f}", g_name)
                g_filename = f"{g_name}{ext}"
                g_image.save(os.path.join(G_PATH, g_filename))

                if 'last_key' not in locals():
                    last_key = (s_F, s_dx, s_I)
                    last_Tg_max = Tg_max
                    last_Tg_min = Tg_min
                else:
                    current_key = (s_F, s_dx, s_I)
                    if current_key == last_key:
                        last_Tg_max = max(last_Tg_max, Tg_max)
                        last_Tg_min = min(last_Tg_min, Tg_min)
                        add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), s_F, s_dx, s_I, 'gap_max (°C)', last_Tg_max)
                        add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), s_F, s_dx, s_I, 'gap_min (°C)', last_Tg_min)
                    else:
                        last_key = current_key
                        last_Tg_max = Tg_max
                        last_Tg_min = Tg_min


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DiGCT Data')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config_data.yml')

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

    # Preprocess the dataset
    data_preprocess(args=args)
    print("Dataset preprocessing is completed.")