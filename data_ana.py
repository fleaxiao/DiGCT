import os
import re
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageDraw

from data_tool import value2graypixel, s2c_distance, s2r_distance, add_t_range, sigmoid


def analytical_model(args):

    # load config file
    DATASET_PATH = args.dataset_path
    S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
    A_PATH = os.path.join(DATASET_PATH, "IGCT", "A")
    G_PATH = os.path.join(DATASET_PATH, "IGCT", "G")
    T_RANGE_PATH = os.path.join(DATASET_PATH, "IGCT", "T_range.csv")
    os.makedirs(A_PATH, exist_ok=True)
    os.makedirs(G_PATH, exist_ok=True)
    IMAGE_SIZE = args.image_size
    
    F_standard = float(args.F_standard)
    alpha = float(args.alpha)
    beta = float(args.beta)

    Tc = float(args.Tc)
    hg = float(args.hg)
    h_air = float(args.h_air)

    Vce = float(args.Vce)
    rce = float(args.rce)
    f = float(args.f)
    Irate = float(args.Irate)
    Eon = float(args.Eon)
    Eoff = float(args.Eoff)

    R_node_1 = float(args.R_node_1)
    R_node_2 = float(args.R_node_2)
    R_node_3 = float(args.R_node_3)
    R_heatsink = float(args.R_heatsink)
    L_heaksink = float(args.L_heatsink)
    R_spacer = float(args.R_spacer)
    L_spacer = float(args.L_spacer)
    R_chip = float(args.R_chip)
    L_chip = float(args.L_chip)

    k_cu = float(args.k_cu)
    k_mo = float(args.k_mo)
    k_si = float(args.k_si)

    A_heatsink = np.pi * R_heatsink ** 2
    A_spacer = np.pi * R_spacer ** 2
    A_chip = np.pi * R_chip ** 2

    Res_heatink_air = 1 / (h_air * A_heatsink)
    Res_heatsink = L_heaksink / (k_cu * A_heatsink)
    Res_heatsink_spacer = 1 / (hg * min(A_heatsink, A_spacer))
    Res_spacer = L_spacer / (k_mo * A_spacer)
    Res_spacer_chip = 1 / (hg * min(A_spacer, A_chip))
    Res_chip = L_chip / (k_si * A_chip)

    R_side = 0.5 * Res_chip + Res_spacer_chip + Res_spacer + Res_heatsink_spacer + Res_heatsink + Res_heatink_air
    R_total = 1 / (1 / R_side + 1 / R_side)

    for filename in tqdm(os.listdir(S_PATH), desc="Processing files"):
        # Extract F and I information from the filename
        parts = filename.replace('.png', '').split('_')
        F = int(parts[1])
        dx = int(parts[3])
        I = int(parts[5])
        angle = int(parts[7])
        Ts_max = float(parts[9])
        Ts_min = float(parts[11])
        
        P_cond = np.sqrt(2) * I * Vce * (1 / (2 * np.pi) + (0.8 * 0.85) / 8) + 2 * (I ** 2) * rce * (1 / 8 + (0.8 * 0.85) / (3 * np.pi))
        P_sw = 1 / (np.sqrt(2) * np.pi) * f * (I / Irate) * (Eon + Eoff)
        P = P_cond + P_sw

        Tj = Tc + P * R_total
        Tnode = Tc + P * R_total * 2/5

        # force and temperature coefficients
        coef_F = sigmoid(alpha * (F / F_standard - 1)) # If F == F_standard, coef_F = 0.5
        coef_T = sigmoid(beta * (Tj / Tc - 1))
        Ta_max = Tj * (1 + (1 - coef_F)/2)
        Ta_min = Tj * (1 - (1 - coef_F)/2)

        Tj_max = (Tj * (1 + (1 - coef_F)/2) - Ta_min) / (Ta_max - Ta_min)
        Tj_min = (Tj * (1 - (1 - coef_F)/2) - Ta_min) / (Ta_max - Ta_min)
        Tnode_max = (Tnode * (1 - (1 - coef_F)/2) * (2 -coef_F) - Ta_min) / (Ta_max - Ta_min)
        Tnode_min = (Tnode * (1 - (1 - coef_F)/2)- Ta_min) / (Ta_max - Ta_min)

        ## analysis image
        Tj_list = list(np.linspace(Tj_max, Tj_min, num=IMAGE_SIZE//2, endpoint=False))
        Tj_list = [value2graypixel(value) for value in Tj_list]
        Tnode_list = list(np.linspace(Tnode_max, Tnode_min, num=IMAGE_SIZE//2, endpoint=False))
        Tnode_list = [value2graypixel(value) for value in Tnode_list]

        a_image = s2c_distance(None, Tj_list, IMAGE_SIZE)
        a_image = s2r_distance(a_image, Tnode_list, IMAGE_SIZE, 0, R_node_1, R_chip)
        a_image = s2r_distance(a_image, Tnode_list, IMAGE_SIZE, R_node_2, R_node_3, R_chip)

        name, ext = os.path.splitext(filename)
        Ta_max_match = re.search(r"Tmax_([\d\.]+)", name)
        if  Ta_max_match:
            a_name = re.sub(r"Tmax_([\d\.]+)", f"Tmax_{Ta_max:.2f}", name)
        Ta_min_match = re.search(r"Tmin_([\d\.]+)", name)
        if  Ta_min_match:
            a_name = re.sub(r"Tmin_([\d\.]+)", f"Tmin_{Ta_min:.2f}", a_name)

        a_filename = f"{a_name}{ext}"
        a_image.save(os.path.join(A_PATH, a_filename))

        add_t_range(T_RANGE_PATH, F, dx, I, 'analysis_max (°C)', Ta_max)
        add_t_range(T_RANGE_PATH, F, dx, I, 'analysis_min (°C)', Ta_min)

        ## gap image
        a_np = np.array(a_image)[:,:,1]
        a_np = a_np.astype(np.float32) / 255.0 * (Ta_max - Ta_min) + Ta_min

        s_np = np.array(Image.open(os.path.join(S_PATH, filename)))[:,:,1]
        s_np = s_np.astype(np.float32) / 255.0 * (Ts_max - Ts_min) + Ts_min

        g_np = s_np - a_np
        Tg_max, Tg_min = np.max(g_np), np.min(g_np)
        g_np = (g_np - Tg_min) / (Tg_max - Tg_min) * 255.0
        g_image = Image.fromarray(g_np.astype(np.uint8), mode='L')
        mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=255)
        g_image.putalpha(mask)
        
        Tg_max_match = re.search(r"Tmax_([\d\.]+)", name)
        if  Tg_max_match:
            g_name = re.sub(r"Tmax_([\d\.]+)", f"Tmax_{Tg_max:.2f}", name)
        Tg_min_match = re.search(r"Tmin_([\d\.]+)", name)
        if  Tg_min_match:
            g_name = re.sub(r"Tmin_([\d\.]+)", f"Tmin_{Tg_min:.2f}", g_name)
        g_filename = f"{g_name}{ext}"
        g_image.save(os.path.join(G_PATH, g_filename))

        Tg_max = np.float64(Tg_max)
        Tg_min = np.float64(Tg_min)

        add_t_range(T_RANGE_PATH, F, dx, I, 'gap_max (°C)', Tg_max)
        add_t_range(T_RANGE_PATH, F, dx, I, 'gap_min (°C)', Tg_min)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Physics-Informed Dataset Preprocessing')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config_ana.yml')

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
    analytical_model(args=args)

    print("Analytical reference is generated.")