import numpy as np

from data_tool import sigmoid


def analytical_model(args, F, dx, I):

    # load config file    
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
    L_node = float(args.L_node)
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

        
    P_cond = np.sqrt(2) * I * Vce * (1 / (2 * np.pi) + (0.8 * 0.85) / 8) + 2 * (I ** 2) * rce * (1 / 8 + (0.8 * 0.85) / (3 * np.pi))
    P_sw = 1 / (np.sqrt(2) * np.pi) * f * (I / Irate) * (Eon + Eoff)
    P = P_cond + P_sw

    Tj = Tc + P * R_total
    Tnode = Tc + P * R_total * (L_chip - L_node) / L_chip

    # force and temperature coefficients
    coef_F = sigmoid(alpha * (F / F_standard - 1)) # If F == F_standard, coef_F = 0.5
    coef_T = sigmoid(beta * (Tj / Tc - 1))
    Tj_max = round(Tj * (1 + (1 - coef_F)/2), 2)
    Tj_min = round(Tj * (1 - (1 - coef_F)/2), 2)

    Tnode_max = round(Tnode * (1 + (1 - coef_F)/2), 2)
    Tnode_min = round(Tnode * (1 - (1 - coef_F)/2), 2)

    return Tj_max, Tj_min, Tnode_max, Tnode_min