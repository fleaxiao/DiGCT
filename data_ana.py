import os
import re
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from matplotlib import cm

from data_tool import value2gray, s2c_distance, s2r_distance, read_t_range, sigmoid


def analytical_model(args):

    # load config file
    DATASET_PATH = args.dataset_path
    S_PATH = os.path.join(DATASET_PATH, "IGCT", "S")
    A_PATH = os.path.join(DATASET_PATH, "IGCT", "A")
    os.makedirs(A_PATH, exist_ok=True)
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
        if filename.endswith(".png"):
            # Extract F and I information from the filename
            parts = filename.split('_')
            F = int(parts[1])
            dx = int(parts[3])
            I = int(parts[5])
            
            P_cond = np.sqrt(2) * I * Vce * (1 / (2 * np.pi) + (0.8 * 0.85) / 8) + 2 * (I ** 2) * rce * (1 / 8 + (0.8 * 0.85) / (3 * np.pi))
            P_sw = 1 / (np.sqrt(2) * np.pi) * f * (I / Irate) * (Eon + Eoff)
            P = P_cond + P_sw

            Tj = Tc + P * R_total
            Tnode = Tc + P * R_total * 2/5

            # force and temperature coefficients
            coef_F = sigmoid(alpha * (F / F_standard - 1)) # If F == F_standard, coef_F = 0.5
            coef_T = sigmoid(beta * (Tj / Tc - 1))
            T_max = Tj * (1 + (1 - coef_F)/2)
            T_min = Tj * (1 - (1 - coef_F)/2)

            Tj_max = (Tj * (1 + (1 - coef_F)/2) - T_min) / (T_max - T_min)
            Tj_min = (Tj * (1 - (1 - coef_F)/2) - T_min) / (T_max - T_min)
            Tnode_max = (Tnode * (1 - (1 - coef_F)/2) * (2 -coef_F) - T_min) / (T_max - T_min)
            Tnode_min = (Tnode * (1 - (1 - coef_F)/2)- T_min) / (T_max - T_min)

            Tj_list = list(np.linspace(Tj_max, Tj_min, num=IMAGE_SIZE//2, endpoint=False))
            Tj_list = [value2gray(value) for value in Tj_list]
            Tnode_list = list(np.linspace(Tnode_max, Tnode_min, num=IMAGE_SIZE//2, endpoint=False))
            Tnode_list = [value2gray(value) for value in Tnode_list]

            image = s2c_distance(None, Tj_list, IMAGE_SIZE)
            image = s2r_distance(image, Tnode_list, IMAGE_SIZE, 0, R_node_1, R_chip)
            image = s2r_distance(image, Tnode_list, IMAGE_SIZE, R_node_2, R_node_3, R_chip)

            name, ext = os.path.splitext(filename)
            T_max_match = re.search(r"Tmax_([\d\.]+)", name)
            if  T_max_match:
                name = re.sub(r"Tmax_([\d\.]+)", f"Tmax_{T_max:.2f}", name)
            T_min_match = re.search(r"Tmin_([\d\.]+)", name)
            if  T_min_match:
                name = re.sub(r"Tmin_([\d\.]+)", f"Tmin_{T_min:.2f}", name)

            output_filename = f"{name}{ext}"
            output_path = os.path.join(A_PATH, output_filename)
            image.save(output_path)


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