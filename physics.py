import os
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageDraw


def physics_informed_model(args):

    # load config file
    DATASET_PATH = args.dataset_path
    S_PATH = os.path.join(DATASET_PATH, "S")
    A_PATH = os.path.join(DATASET_PATH, "A")
    os.makedirs(A_PATH, exist_ok=True)
    IMAGE_SIZE = args.image_size
    

    T_max = float(args.T_max)
    T_min = float(args.T_min)
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
        if filename.startswith("surface") and filename.endswith(".png"):
            # Extract F and I information from the filename
            parts = filename.split('_')
            F = int(parts[2])
            I = int(parts[6])
            P_cond = np.sqrt(2) * I * Vce * (1 / (2 * np.pi) + (0.8 * 0.85) / 8) + 2 * (I ** 2) * rce * (1 / 8 + (0.8 * 0.85) / (3 * np.pi))
            P_sw = 1 / (np.sqrt(2) * np.pi) * f * (I / Irate) * (Eon + Eoff)
            P = P_cond + P_sw

            Tj = Tc + P * R_total
            Tnode = Tc + P * R_node_1 * 2/5
            Tj_std = (Tj - T_min) / (T_max - T_min)
            Tnode_std = (Tnode - T_min) / (T_max - T_min)

            image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            Tj_color = (int(Tj_std * 255), 0, int((1 - Tj_std) * 255), 255)
            Tnode_color = (int(Tnode_std * 255), 0, int((1 - Tnode_std) * 255), 255)
            draw.ellipse((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=Tj_color)
            draw.ellipse((IMAGE_SIZE/2 - (R_node_3/R_chip * IMAGE_SIZE)/2, IMAGE_SIZE/2 - (R_node_3/R_chip * IMAGE_SIZE)/2, IMAGE_SIZE/2 + (R_node_3/R_chip * IMAGE_SIZE)/2, IMAGE_SIZE/2 + (R_node_3/R_chip * IMAGE_SIZE)/2), fill=Tnode_color)
            draw.ellipse((IMAGE_SIZE/2 - (R_node_2/R_chip * IMAGE_SIZE)/2, IMAGE_SIZE/2 - (R_node_2/R_chip * IMAGE_SIZE)/2, IMAGE_SIZE/2 + (R_node_2/R_chip * IMAGE_SIZE)/2, IMAGE_SIZE/2 + (R_node_2/R_chip * IMAGE_SIZE)/2), fill=Tj_color)
            draw.ellipse((IMAGE_SIZE/2 - (R_node_1/R_chip * IMAGE_SIZE)/2, (IMAGE_SIZE/2 - (R_node_1/R_chip * IMAGE_SIZE)/2), (IMAGE_SIZE/2 + (R_node_1/R_chip * IMAGE_SIZE)/2), (IMAGE_SIZE/2 + (R_node_1/R_chip * IMAGE_SIZE)/2)), fill=Tnode_color)

            filename = filename.replace("surface", "Analysis")
            output_path = os.path.join(A_PATH, filename)
            image.save(output_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Physics-Informed Dataset Preprocessing')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/physics.yml')

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
    physics_informed_model(args=args)

    print("Physics-Informed Dataset preprocessing is completed.")


