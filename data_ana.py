import argparse
import yaml
import numpy as np

from data_tool import sigmoid


def calculate_heat_transfer_coefficient_flat_plate(vf, L, k_water, Pr_water, mu_water, rho_water):
    Re_L = rho_water * vf * L / mu_water
    
    if Re_L <= 5e5:
        Nu = 0.3387 * (Pr_water**(1/3)) * (Re_L**(1/2)) / ((1 + (0.0468/Pr_water)**(2/3))**(1/4))
    else:
        Nu = (0.037 * (Re_L**(4/5)) - 871) * (Pr_water**(1/3))
    
    h = Nu * k_water / L
    
    return h, Re_L, Nu

def analytical_model(args, F, theta, vf, I):

    Tc = float(args.Tc)
    h_g = float(args.h_g)

    Vce = float(args.Vce)
    rce = float(args.rce)
    f = float(args.f)
    Irate = float(args.Irate)
    Eon = float(args.Eon)
    Eoff = float(args.Eoff)

    R_node_1 = float(args.R_node_1)
    R_node_2 = float(args.R_node_2)
    R_node_3 = float(args.R_node_3)
    L_node = float(args.L_node)
    R_heatsink = float(args.R_heatsink)
    L_heatsink = float(args.L_heatsink)
    R_spacer = float(args.R_spacer)
    L_spacer = float(args.L_spacer)
    R_chip = float(args.R_chip)
    L_chip = float(args.L_chip)

    k_cu = float(args.k_cu)
    k_mo = float(args.k_mo)
    k_si = float(args.k_si)

    k_water = float(args.k_water)
    Pr_water = float(args.Pr_water)
    mu_water = float(args.mu_water)
    rho_water = float(args.rho_water)

    A_heatsink = np.pi * R_heatsink ** 2
    A_spacer = np.pi * R_spacer ** 2
    A_chip = np.pi * R_chip ** 2

    Res_heatsink = L_heatsink / (k_cu * A_heatsink)
    Res_heatsink_spacer = 1 / (h_g * min(A_heatsink, A_spacer))
    Res_spacer = L_spacer / (k_mo * A_spacer)
    Res_spacer_chip = 1 / (h_g * min(A_spacer, A_chip))
    Res_chip = L_chip / (k_si * A_chip)

    L_characteristic = 0.01
    h_db, Re_L, Nu = calculate_heat_transfer_coefficient_flat_plate(vf, L_characteristic, k_water, Pr_water, mu_water, rho_water)

    Res_heatsink_water = 1 / (h_db * A_heatsink)

    R_side = 0.5 * Res_chip + Res_spacer + Res_heatsink + Res_heatsink_water + Res_spacer_chip + Res_heatsink_spacer
    R_total = 1 / (1 / R_side + 1 / R_side)

    P_cond = np.sqrt(2) * I * Vce * (1 / (2 * np.pi) + (0.8 * 0.85) / 8) + 2 * (I ** 2) * rce * (1 / 8 + (0.8 * 0.85) / (3 * np.pi))
    P_sw = 1 / np.pi * f * (I / Irate) * (Eon + Eoff)
    P = P_cond + P_sw

    Tj = Tc + P * R_total
    Tj = round(Tj, 2)

    return Tj

if __name__ == "__main__":
    p = argparse.ArgumentParser(description='DiGCT Data')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config_data.yml')

    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    Tj = analytical_model(args, F=35, theta=0, vf=1, I=1000)
    print("Estimated junction temperature (Tj):", Tj)