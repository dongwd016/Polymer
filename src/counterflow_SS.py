import os
import json

import cantera as ct
import numpy as np
import scipy
from scipy import constants as cst

root_dir = "D:/DocumentAll/Research"
work_dir = "{}/2-Polymer".format(root_dir)


def json_convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "tolist"):
        return {"$nparray": obj.tolist()}


def json_deconvert(obj):
    if len(obj) == 1:
        key, value = next(iter(obj.items()))
        if key == "$nparray":
            return np.array(value)
    return obj


def bisection(rhs, left, right, val_left, val_right, tol=1e-10):
    if val_left * val_right > 0:
        # print('Bisection same sign!')
        # return 0
        raise Exception("Bisection same sign!")

    mid = (left + right) / 2
    val_mid = rhs(mid)
    if val_mid == 0:
        return mid

    if val_mid * val_left < 0:
        right = mid
        val_right = val_mid
    else:
        left = mid
        val_left = val_mid

    if right - left < tol:
        return mid
    else:
        return bisection(rhs, left, right, val_left, val_right, tol)


case_ind = 1
case_folder = "{}/output/counterflow/Case{}".format(work_dir, case_ind)
if not os.path.isdir(case_folder):
    os.makedirs(case_folder)

r_evap = 1.0  # (sum_i{nu_i*H_vap_i} + dH) / dH
inui = 1.0  # sum_i{i*nu_i}
T0 = 300  # K, initial temperature of the polymer
T_melt = 438  # K, POM melting point
k = 0.14  # W/m·K, liquid phase thermal conductivity
rho = 1.2e3  # kg/m3, liquid phase density
MW0 = 30e-3  # kg/mol, CH2O molecular weight
MW0_t = MW0 * inui
lumped_A = 2 * 1.8e13  # 1/s
lumped_Ea = 30e3 * cst.calorie  # J/mol
dH = 56e3  # J/mol, heat absorbed by beta scission
dH_t = dH * r_evap
N = 3333  # number of polymer degree
MW = N * MW0  # kg/mol, molecular weight of POM
lh = 150e3  # J/kg, latent heat of POM melting
cp = 35 / MW0  # J/kg·K, specific heat
mu = 1 + cp * (T_melt - T0) / lh
lh_t = lh * mu

At = lumped_A * rho / (k * MW) * dH_t
Te = lumped_Ea / cst.gas_constant
g = lambda u: scipy.special.expi(-1 / u) + u * np.exp(-1 / u)
T_ratio = 0.5

gas = ct.Solution("{}/data/FFCM2_CH2O.yaml".format(work_dir))
width = 1e-2
f = ct.CounterflowDiffusionFlame(gas, width=width)
f.set_refine_criteria(ratio=2, slope=0.1, curve=0.1, prune=0.01)

fuel = "CH2O"
solid_sp = "POM"
p = ct.one_atm
comp_o = "O2:1"
comp_f = "{}:1".format(fuel)

tin_o = T0
f.fuel_inlet.X = comp_f
f.oxidizer_inlet.X = comp_o
f.oxidizer_inlet.T = tin_o
gas.TPX = tin_o, p, comp_o
rho_o = gas.density_mass

cond_ind = 1
folder = "{}/cond{}".format(case_folder, cond_ind)
if not os.path.isdir(folder):
    os.makedirs(folder)

mdot_o = 1
f.oxidizer_inlet.mdot = mdot_o
vel_o = mdot_o / rho_o


def get_props(Ts):
    tin_f = Ts
    gas.TPX = tin_f, p, comp_f
    rho_f = gas.density_mass
    f.fuel_inlet.T = tin_f

    T_mid = T_ratio * Ts + (1 - T_ratio) * T_melt
    q0 = np.sqrt(
        At
        * Te
        * k**2
        * (g(T_mid / Te) - g(Ts / Te))
        / (0.5 * ((lh + cp * (T_mid - T0)) / (lh + cp * (Ts - T0) + dH_t / MW0_t)) ** 2 - 0.5 - cp * (T_mid - Ts) / (lh + cp * (Ts - T0) + dH_t / MW0_t))
    )
    mdot_f = q0 / (lh + cp * (Ts - T0) + dH_t / MW0_t)
    vel_f = mdot_f / rho_f
    rb = mdot_f / rho
    f.fuel_inlet.mdot = mdot_f
    f.solve(loglevel=0, auto=True)

    k_arr = f.thermal_conductivity
    temp_arr = f.T
    x_arr = f.grid
    # Get gas-liquid interface heat flux
    dx0 = x_arr[1] - x_arr[0]
    dx1 = x_arr[2] - x_arr[1]
    dTdx0 = -(2 * dx0 + dx1) / (dx0 * (dx0 + dx1)) * temp_arr[0] + (dx0 + dx1) / (dx0 * dx1) * temp_arr[1] - dx0 / (dx1 * (dx0 + dx1)) * temp_arr[2]
    q = k_arr[0] * dTdx0  # heat flux to liquid (-x direction)
    return q, q0, vel_f, rb


def rhs(Ts):
    q, q0, vel_f, rb = get_props(Ts)
    if np.max(f.T) < Ts + 10:
        return 1.0
    else:
        return q - q0


left = 500
right = 800

case_dict = {
    "case_ind": case_ind,
    "cond_ind": cond_ind,
    "solid_sp": solid_sp,
    "fuel": fuel,
    "p": p,
    "tin_o": tin_o,
    "comp_o": comp_o,
    "comp_f": comp_f,
    "width": width,
    "vel_o": vel_o,
    "left": left,
    "right": right,
    "T_ratio": T_ratio,
}
json.dump(case_dict, open("{}/case_dict.json".format(folder), "w"), indent=4, default=json_convert)

Ts = bisection(rhs, left, right, rhs(left), rhs(right), tol=1e-4)
q, q0, vel_f, rb = get_props(Ts)
case_dict["Ts"] = Ts
case_dict["vel_f"] = vel_f
case_dict["rb"] = rb
json.dump(case_dict, open("{}/case_dict.json".format(folder), "w"), indent=4, default=json_convert)
f.save("{}/solution.h5".format(folder), name="solution", overwrite=True)
