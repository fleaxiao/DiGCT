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

    TABLE = args.table
    SURFACE = args.surface
    SIDE = args.side
    G = args.G
    I_option = args.I_option

    R_node_1 = float(args.R_node_1)
    R_node_2 = float(args.R_node_2)
    R_node_3 = float(args.R_node_3)
    L_node = float(args.L_node)
    R_chip = float(args.R_chip)
    L_chip = float(args.L_chip)

    F_min = args.F_min
    F_max = args.F_max

    surface_start = 378
    surface_end = 2526
    side_start = 132
    side_end = 2772

    surface_center = surface_start + (surface_end - surface_start) // 2
    surface_radius = (surface_end - surface_start) // 2
    side_length = side_end - side_start

    # Process csv file
    if TABLE:
        t_range = pd.read_csv(os.path.join(DATA_PATH, "T_range.csv"), header=4)
        t_range.rename(columns={t_range.columns[0]: "F (kN)"}, inplace=True)
        t_range["F (kN)"] = t_range["F (kN)"].abs()
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

    # Process surface images
    if SURFACE:
        surface_files = [f for f in os.listdir(DATA_PATH) if f.startswith("surface") and f.endswith(".png") and not f.endswith("_0.png")]
        S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
        os.makedirs(S_PATH, exist_ok=True)
        for filename in tqdm(surface_files, desc="Processing surface images"):
            if filename.startswith("surface") and filename.endswith(".png") and not filename.endswith("_0.png"):
                parts = filename.split('_')
                F = int(abs(float(parts[2])) / 1000)
                theta = int(float(parts[4]))
                vf = float(parts[6])
                I = int(os.path.splitext(parts[8])[0])
                surface_max, surface_min, _, _ = read_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, theta, vf, I)

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
                        rotated_img = image_add_0(rotated_img, margin=MARGIN)
                        rotated_img = image_add_mask(rotated_img, margin=MARGIN)

                        name, ext = os.path.splitext(filename)
                        f_number_match = re.search(r"F_(-?\d+)", name)
                        if f_number_match:
                            f_number = int(abs(int(f_number_match.group(1))) / 1000)
                            name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                        theta_number_match = re.search(r"theta_([\d\.]+)", name)
                        if theta_number_match:
                            theta_number = int(float(theta_number_match.group(1)))
                            name = re.sub(r"theta_([\d\.]+)", f"theta_{theta_number}", name)
                        name = name.replace("surface_", "")
                        output_filename = f"{name}_angle_{angle}_Tmax_{surface_max}_Tmin_{surface_min}{ext}"
                        output_path = os.path.join(S_PATH, output_filename)
                        rotated_img.save(output_path)
    
    # Process side images
    if SIDE:
        side_files = [f for f in os.listdir(DATA_PATH) if f.startswith("side") and f.endswith(".png") and not f.endswith("_0.png")]
        for filename in tqdm(side_files, desc="Processing side images"):
            parts = filename.split('_')
            F = int(abs(float(parts[2])) / 1000)
            theta = int(float(parts[4]))
            vf = float(parts[6])
            I = int(os.path.splitext(parts[8])[0])
            _, _, side_max, side_min = read_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, theta, vf, I)

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
                        f_number = int(abs(int(f_number_match.group(1))) / 1000)
                        name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                    theta_number_match = re.search(r"theta_([\d\.]+)", name)
                    if theta_number_match:
                        theta_number = int(float(theta_number_match.group(1)))
                        name = re.sub(r"theta_([\d\.]+)", f"theta_{theta_number}", name)
                    name = name.replace("side_", "")
                    output_filename = f"{name}_angle_{angle}_Tmax_{side_max}_Tmin_{side_min}{ext}"
                    output_path = os.path.join(L_PATH, output_filename)
                    img_rotated.save(output_path)

                ## l2s side image w/o analysis model
                match I_option:
                    case "L2S":
                        L2S_PATH = os.path.join(DATASET_PATH, "IGCT", "L2S")
                        os.makedirs(L2S_PATH, exist_ok=True)

                        img_o = Image.new("L", (len(pixels), 1), 0)
                        img_o.putdata(pixels)
                        img_o = img_o.resize((IMAGE_SIZE, 1), Image.Resampling.NEAREST)
                        l2s_list = list(img_o.getdata())
                        l2s_list.extend(reversed(l2s_list))

                        l2s_image = s2c_angle(None, l2s_list, 255, None, IMAGE_SIZE)
                        l2s_image = l2s_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

                        for angle in range(0, 360, ANGLE_STEP):
                            rotated_img = l2s_image.rotate(angle, expand=False)
                            rotated_img = image_add_0(rotated_img, margin=MARGIN)
                            rotated_img = image_add_mask(rotated_img, margin=MARGIN)

                            name, ext = os.path.splitext(filename)
                            f_number_match = re.search(r"F_(-?\d+)", name)
                            if f_number_match:
                                f_number = int(abs(int(f_number_match.group(1))) / 1000)
                                name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                            theta_number_match = re.search(r"theta_([\d\.]+)", name)
                            if theta_number_match:
                                theta_number = int(float(theta_number_match.group(1)))
                                name = re.sub(r"theta_([\d\.]+)", f"theta_{theta_number}", name)
                            name = name.replace("side_", "")
                            output_filename = f"{name}_angle_{angle}_Tmax_{side_max}_Tmin_{side_min}{ext}"
                            output_path = os.path.join(L2S_PATH, output_filename)
                            rotated_img.save(output_path)

                    ## p2s side image w/o analysis model
                    case "P2S":
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
                            p2s_image = s2c_angle(None, p2s_list, np.mean(p2s_list), None, IMAGE_SIZE)
                            p2s_image = image_add_0(p2s_image, margin=MARGIN)
                            p2s_image = image_add_mask(p2s_image, margin=MARGIN)

                            name, ext = os.path.splitext(filename)
                            f_number_match = re.search(r"F_(-?\d+)", name)
                            if f_number_match:
                                f_number = int(abs(int(f_number_match.group(1))) / 1000)
                                name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                            theta_number_match = re.search(r"theta_([\d\.]+)", name)
                            if theta_number_match:
                                theta_number = int(float(theta_number_match.group(1)))
                                name = re.sub(r"theta_([\d\.]+)", f"theta_{theta_number}", name)
                            name = name.replace("side_", "")
                            output_filename = f"{name}_angle_{angle}_Tmax_{side_max}_Tmin_{side_min}{ext}"
                            output_path = os.path.join(P2S_PATH, output_filename)
                            p2s_image.save(output_path)

                    ## pa2s side image w analysis model
                    case "PA2S":
                        PA2S_PATH = os.path.join(DATASET_PATH, "IGCT", "PA2S")
                        os.makedirs(PA2S_PATH, exist_ok=True)

                        Tj = analytical_model(args, F, theta, vf, I)
                        Ta_max = max(Tj, side_max)
                        Ta_min = min(Tj, side_min)
                        Tj_pixel = int((Tj - Ta_min) / (Ta_max - Ta_min) * 255.0)

                        pixels_extend = pixels.copy()
                        reversed_pixels = list(reversed(pixels))
                        pixels_extend.extend(reversed_pixels)
                        T_extend = np.array(pixels_extend) * (side_max - side_min) / 255.0 + side_min

                        k = (F - F_min) / (F_max - F_min)

                        for angle in range(0, 360, ANGLE_STEP):
                            start_pixel = int(2 * angle * side_length / 360)
                            interval = side_length // 4
                            p2s_list = np.array([T_extend[(start_pixel + i * interval) % len(T_extend)] for i in range(8)]) 
                            p2s_list = (p2s_list - Ta_min) / (Ta_max - Ta_min) * 255.0
                            p2s_list = smooth_interpolation(p2s_list, IMAGE_SIZE + 1)
                            p2s_image = s2c_angle(None, p2s_list, Tj_pixel, k, IMAGE_SIZE)
                            # p2s_image = s2r_angle(p2s_image, p2s_list, Tj_pixel * (L_chip - L_node) / L_chip, IMAGE_SIZE, 0, R_node_1, R_chip)
                            # p2s_image = s2r_angle(p2s_image, p2s_list, Tj_pixel * (L_chip - L_node) / L_chip, IMAGE_SIZE, R_node_2, R_node_3, R_chip)
                            p2s_image = image_add_0(p2s_image, margin=MARGIN)
                            p2s_image = image_add_mask(p2s_image, margin=MARGIN)

                            name, ext = os.path.splitext(filename)
                            f_number_match = re.search(r"F_(-?\d+)", name)
                            if f_number_match:
                                f_number = int(abs(int(f_number_match.group(1))) / 1000)
                                name = re.sub(r"F_(-?\d+)", f"F_{f_number}", name)
                            theta_number_match = re.search(r"theta_([\d\.]+)", name)
                            if theta_number_match:
                                theta_number = int(float(theta_number_match.group(1)))
                                name = re.sub(r"theta_([\d\.]+)", f"theta_{theta_number}", name)
                            name = name.replace("side_", "")
                            output_filename = f"{name}_angle_{angle}_Tmax_{Ta_max}_Tmin_{Ta_min}{ext}"
                            output_path = os.path.join(PA2S_PATH, output_filename)
                            p2s_image.save(output_path)

                            add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, theta, vf, I, 'analysis_max (°C)', Ta_max)
                            add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), F, theta, vf, I, 'analysis_min (°C)', Ta_min)

    # Process gap images
    if G:
        S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
        if not os.path.exists(S_PATH):
            raise FileNotFoundError(f"Surface images directory {S_PATH} does not exist. Please process surface images first.")
        if not os.listdir(S_PATH):
            raise ValueError(f"Surface images directory {S_PATH} is empty. Please process surface images first.")

        s_files = sorted([f for f in os.listdir(S_PATH) if f.endswith('.png')])
        match I_option:
            case "L2S":
                E_PATH = os.path.join(DATASET_PATH, "IGCT", "L2S")
                e_files = sorted([f for f in os.listdir(E_PATH) if f.endswith('.png')])
                G_PATH = os.path.join(DATASET_PATH, "IGCT", "G_L2S")
                os.makedirs(G_PATH, exist_ok=True)
            case "P2S":
                E_PATH = os.path.join(DATASET_PATH, "IGCT", "P2S")
                e_files = sorted([f for f in os.listdir(E_PATH) if f.endswith('.png')])
                G_PATH = os.path.join(DATASET_PATH, "IGCT", "G_P2S")
                os.makedirs(G_PATH, exist_ok=True)
            case "PA2S":
                E_PATH = os.path.join(DATASET_PATH, "IGCT", "PA2S")
                e_files = sorted([f for f in os.listdir(E_PATH) if f.endswith('.png')])
                G_PATH = os.path.join(DATASET_PATH, "IGCT", "G_PA2S")
                os.makedirs(G_PATH, exist_ok=True)
    
        for s_file, e_file in tqdm(zip(s_files, e_files), total=min(len(s_files), len(e_files)), desc="Processing gap images"):
            s_name, ext = os.path.splitext(s_file)
            s_img_path = os.path.join(S_PATH, s_file)
            e_img_path = os.path.join(E_PATH, e_file)
            with Image.open(s_img_path).convert("L") as s_img, Image.open(e_img_path).convert("L") as e_img:
                s_F, s_theta, s_vf, s_I, s_angle, s_Tmax, s_Tmin = extract_label(s_img_path)
                e_F, e_theta, e_vf, e_I, e_angle, e_Tmax, e_Tmin = extract_label(e_img_path)
                if not (s_F == e_F) or not (s_theta == e_theta) or not (s_vf == e_vf) or not (s_I == e_I) or not (s_angle == e_angle):
                    raise ValueError(
                        f"Parameters must match for gap image:\n"
                        f"  F values - Surface: {s_F}, PA2S: {e_F}\n"
                        f"  theta values - Surface: {s_theta}, PA2S: {e_theta}\n"
                        f"  vf values - Surface: {s_vf}, PA2S: {e_vf}\n"
                        f"  I values - Surface: {s_I}, PA2S: {e_I}\n"
                        f"  angle values - Surface: {s_angle}, PA2S: {e_angle}"
                    )

                s_np = np.array(s_img)
                s_np = s_np.astype(np.float32) / 255.0 * (s_Tmax - s_Tmin) + s_Tmin

                e_np = np.array(e_img)
                e_np = e_np.astype(np.float32) / 255.0 * (e_Tmax - e_Tmin) + e_Tmin

                g_np = s_np - e_np
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
                    last_key = (s_F, s_theta, s_I)
                    last_Tg_max = Tg_max
                    last_Tg_min = Tg_min
                else:
                    current_key = (s_F, s_theta, s_I)
                    if current_key == last_key:
                        last_Tg_max = max(last_Tg_max, Tg_max)
                        last_Tg_min = min(last_Tg_min, Tg_min)
                        add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), s_F, s_theta, s_vf, s_I, 'gap_max (°C)', last_Tg_max)
                        add_t_range(os.path.join(DATASET_PATH, "IGCT", "T_range.csv"), s_F, s_theta, s_vf, s_I, 'gap_min (°C)', last_Tg_min)
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