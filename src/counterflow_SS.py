import os
import json

import cantera as ct
import numpy as np
import scipy
from scipy import constants as cst
from tqdm.auto import tqdm

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


def get_q(f, diffusion_term=True, soret_effect=True, radiation_term=True, ee=1.0):
    def _fwd_grad_nonuniform(z0, z1, z2, f0, f1, f2):
        dx0 = z1 - z0
        dx1 = z2 - z1
        c0 = -(2 * dx0 + dx1) / (dx0 * (dx0 + dx1))
        c1 = (dx0 + dx1) / (dx0 * dx1)
        c2 = -dx0 / (dx1 * (dx0 + dx1))
        return c0 * f0 + c1 * f1 + c2 * f2

    def _pick_nodes(zsize, side):
        if side.lower().startswith("fuel"):
            return 0, 1, 2, +1
        else:
            return zsize - 1, zsize - 2, zsize - 3, -1

    def _solve_stefan_maxwell_N(gas, dXdz, xDT_dlnTdz=None):
        X = gas.X.copy()
        K = gas.n_species
        ctot = np.sum(gas.concentrations)  # kmol/m3
        Dij = gas.binary_diff_coeffs  # (K,K) m2/s

        A = np.zeros((K, K))
        b = -ctot * dXdz.copy()
        if xDT_dlnTdz is not None:
            b -= ctot * xDT_dlnTdz

        for i in range(K):
            s = 0.0
            for j in range(K):
                if i == j:
                    continue
                A[i, j] = -X[i] / Dij[i, j]
                s += X[j] / Dij[i, j]
            A[i, i] = s

        A[-1, :] = 1.0
        b[-1] = 0.0

        N = np.linalg.solve(A, b)  # kmol/m2/s
        return N

    gas = f.gas
    z, T, Y, rho, P = f.grid, f.T, f.Y, f.density_mass, f.P
    i0, i1, i2, sgn = _pick_nodes(z.size, "fuel")

    dTdz = _fwd_grad_nonuniform(z[i0], z[i1], z[i2], T[i0], T[i1], T[i2]) * sgn
    dYdz = _fwd_grad_nonuniform(z[i0], z[i1], z[i2], Y[:, i0], Y[:, i1], Y[:, i2]) * sgn

    gas.TPY = T[i0], P, Y[:, i0]
    lam = gas.thermal_conductivity
    W = gas.molecular_weights
    h_mass = gas.partial_molar_enthalpies / W

    if f.transport_model == "mixture-averaged":
        Dmix = gas.mix_diff_coeffs_mass

        J_mix_raw = -rho[i0] * Dmix * dYdz
        if f.soret_enabled and soret_effect:
            DT = gas.thermal_diff_coeffs  # m2/s
            J_mix_raw += -rho[i0] * Y[:, i0] * DT * (dTdz / T[i0])
        J_mix = J_mix_raw - Y[:, i0] * np.sum(J_mix_raw)

        q_cond = -lam * dTdz
        q_species = float(np.sum(h_mass * J_mix))

    elif f.transport_model == "multicomponent":
        gas.TPY = T[i0], P, Y[:, i0]
        X0 = gas.X.copy()
        gas.TPY = T[i1], P, Y[:, i1]
        X1 = gas.X.copy()
        gas.TPY = T[i2], P, Y[:, i2]
        X2 = gas.X.copy()
        dXdz = _fwd_grad_nonuniform(z[i0], z[i1], z[i2], X0, X1, X2) * sgn
        Y0 = Y[:, i0]

        gas.TPX = T[i0], P, X0
        if f.soret_enabled and soret_effect:
            DT_mc = gas.thermal_diff_coeffs  # m2/s]
            xDT_dlnTdz = X0 * DT_mc * (dTdz / T[i0])
        else:
            xDT_dlnTdz = None

        N_mc = _solve_stefan_maxwell_N(gas, dXdz, xDT_dlnTdz=xDT_dlnTdz)  # kmol/m^2/s
        J_mc = W * N_mc  # kg/m2/s
        J_mc -= Y0 * np.sum(J_mc)

        q_cond = -lam * dTdz
        q_species = float(np.sum(h_mass * J_mc))
    else:
        raise Exception("Unknown transport model: {}!".format(f.transport_model))

    if f.radiation_enabled:
        arg = np.argmax(f.heat_release_rate)
        q_rad = -cst.sigma * ee * (f.T[arg] ** 4 - f.T[0] ** 4)
    else:
        q_rad = 0.0

    # treat into surface as positive
    q_cond = -q_cond
    q_species = -q_species
    q_rad = -q_rad
    q_total = q_cond
    if diffusion_term:
        q_total += q_species
    if radiation_term:
        q_total += q_rad

    return q_total, {"q_total": q_total, "q_cond": q_cond, "q_species": q_species, "q_rad": q_rad}


def bisection(rhs, left, right, val_left, val_right, tol=1e-10):
    if val_left * val_right > 0:
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
soret_enabled = True
transport_model = "multicomponent"
radiation_enabled = True
diffusion_term = True
soret_effect = True
radiation_term = False
ee = 1.0

case_folder = "{}/output/counterflow/Case{}".format(work_dir, case_ind)
if not os.path.isdir(case_folder):
    os.makedirs(case_folder)

set_ind = 1
fuel = "CH2O"
solid_sp = "POM"
p = 0.1e6
comp_o = "O2:1"
comp_f = "{}:1".format(fuel)
width = 4.3e-3
x_var_name = "vel_o"
x_var_arr = np.arange(10, 111, 10) * 1e-2  # m/s


r_evap = 1.0  # (sum_i{nu_i*H_vap_i} + dH) / dH
inui = 1.0  # sum_i{i*nu_i}
T0 = 298.15  # K, initial temperature of the polymer
T_melt = 438  # K, POM melting point
k = 0.14  # W/m·K, liquid phase thermal conductivity
rho = 1.2e3  # kg/m3, liquid phase density
rho_s = 1.41e3  # kg/m3, solid phase density
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
f = ct.CounterflowDiffusionFlame(gas, width=width)
f.set_refine_criteria(ratio=2, slope=0.1, curve=0.1, prune=0.01)
f.soret_enabled = soret_enabled
f.transport_model = transport_model
f.radiation_enabled = radiation_enabled

tin_o = T0
f.fuel_inlet.X = comp_f
f.oxidizer_inlet.X = comp_o
f.oxidizer_inlet.T = tin_o
gas.TPX = tin_o, p, comp_o
rho_o = gas.density_mass

for cond_ind, x_var in enumerate(tqdm(x_var_arr)):
    folder = "{}/set{}/cond{}".format(case_folder, set_ind, cond_ind)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    print(folder)

    if x_var_name == "vel_o":
        vel_o = x_var
    elif x_var_name == "p_flux":
        p_flux = x_var
        vel_o = np.sqrt(p_flux / rho_o)
    else:
        raise Exception("Unknown x_var_name: {}!".format(x_var_name))
    mdot_o = vel_o * rho_o
    f.oxidizer_inlet.mdot = mdot_o

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

        q, q_dict = get_q(f, diffusion_term=diffusion_term, soret_effect=soret_effect, radiation_term=radiation_term, ee=ee)

        return q, q0, vel_f, rb, q_dict

    def rhs(Ts):
        q, q0, _, _, _ = get_props(Ts)
        if np.max(f.T) < Ts + 10:
            return 1.0
        else:
            return q - q0

    left = 500
    right = 800

    case_dict = {
        "case_ind": case_ind,
        "set_ind": set_ind,
        "cond_ind": cond_ind,
        "solid_sp": solid_sp,
        "fuel": fuel,
        "p": p,
        "tin_o": tin_o,
        "comp_o": comp_o,
        "comp_f": comp_f,
        "width": width,
        "x_var_name": x_var_name,
        "x_var": x_var,
        "vel_o": vel_o,
        "left": left,
        "right": right,
        "T_ratio": T_ratio,
    }
    json.dump(case_dict, open("{}/case_dict.json".format(folder), "w"), indent=4, default=json_convert)

    Ts = bisection(rhs, left, right, rhs(left), rhs(right), tol=1e-4)
    q, q0, vel_f, rb, q_dict = get_props(Ts)
    case_dict.update(q_dict)
    case_dict["q"] = q
    case_dict["Ts"] = Ts
    case_dict["vel_f"] = vel_f
    case_dict["rb"] = rb
    json.dump(case_dict, open("{}/case_dict.json".format(folder), "w"), indent=4, default=json_convert)
    f.save("{}/solution.h5".format(folder), name="solution", overwrite=True)
