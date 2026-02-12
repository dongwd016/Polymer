import itertools
import os
import json

import numpy as np
import pandas as pd
import scipy
from scipy import constants as cst
from tqdm.auto import tqdm
from tqdm.contrib import tzip
import matplotlib.pyplot as plt
import matplotlib.animation

plt.style.use("default")
plt.rc("figure", figsize=[5, 5])
plt.rc("font", size=14, family="Arial")
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=12)
plt.rc("xtick", labelsize=11)
plt.rc("ytick", labelsize=11)
plt.rc("lines", linewidth=2)
# plt.rcParams["animation.html"] = "jshtml"
# plt.rcParams["animation.ffmpeg_path"] = "C:/Program Files/ffmpeg/bin/ffmpeg.exe"
# plt.rcParams["animation.embed_limit"] = 50
mu_sb = "\u03bc"
deg_sb = "\u00b0"

root_dir = "D:/DocumentAll/Research"
work_dir = "{}/2-Polymer".format(root_dir)


# work_dir = '.'


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


def advance_ee(dy_dt, t, y, dt):
    return y + dy_dt(t, y) * dt


def advance_rk4(dy_dt, t, y, dt):
    k1 = dt * dy_dt(t, y)
    k2 = dt * dy_dt(t + dt / 2, y + k1 / 2)
    k3 = dt * dy_dt(t + dt / 2, y + k2 / 2)
    k4 = dt * dy_dt(t + dt, y + k3)
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y + k


def advance_rk4_ln(dy_dt, t, y, dt):
    k1 = dt * dy_dt(t, y)
    k2 = dt * dy_dt(t + dt / 2, np.exp(np.log(y) + k1 / 2))
    k3 = dt * dy_dt(t + dt / 2, np.exp(np.log(y) + k2 / 2))
    k4 = dt * dy_dt(t + dt, np.exp(np.log(y) + k3))
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.exp(np.log(y) + k)


class Polymer:
    def __init__(self):
        self.save_list = [
            "folder",
            "T0",
            "q0",
            "q0_func_str",
            "qb",
            "k_s",
            "k_l",
            "rho_s",
            "rho_l",
            "MW0",
            "cv",
            "cp",
            "A_beta",
            "Ea",
            "dH",
            "MW",
            "gamma",
            "lh",
            "T_melt",
            "slope_Tb",
            "impurity",
            "boundary_condition",
            "L",
            "t_end",
            "Nx",
            "t_num",
            "t_store",
            "Nt",
            "dt",
            "dx",
            "cfl",
            "db_path",
            "sp_name_list",
            "Ns",
            "x_reaction_str",
            "eta",
            "P",
            "lumped_A",
            "lumped_Ea",
            "t_end",
            "t_num",
            "temp_control",
            "n_threshold",
            "diffusion_coefficient",
            "N",
            "D",
            "phase_equilibrium",
            "is_H_vap",
            "min_interval",
            "check_point_step",
        ]
        self.save_npy_list = ["x_arr", "t_arr", "t_arg_arr", "t_store_arr"]
        self.result_list = ["T_mat", "phase_mat", "dL_arr", "fp_mat", "f_ten", "Ei_ten", "h_mat"]
        self.cur_list = ["T_arr", "phase_arr", "dL", "fp_arr", "f_mat", "Ei_mat", "h_arr"]
        self.check_point_list = ["check_point_ind", "check_point_ti"]

        self.folder = None

        # Transient related (heat transfer, non-uniform temperature)
        self.T0 = None  # K
        self.q0 = None  # W/m2, heat flux from gas phase
        self.q0_func = None  # W/m2, heat flux from gas phase as a function of time
        self.q0_func_str = None  # string of q0_func to store in case_dict
        self.qb = None  # W/m2, heat flux from the bottom
        self.k_s = None  # W/m·K, solid phase thermal conductivity
        self.k_l = None  # W/m·K, liquid phase thermal conductivity
        self.rho_s = None  # kg/m3, solid phase density
        self.rho_l = None  # kg/m3, liquid phase density
        self.MW0 = None  # kg/mol, CH2O molecular weight
        self.cv = None  # J/kg·K
        self.cp = None  # J/kg·K
        self.A_beta = None  # 1/s
        self.Ea = None  # J/mol
        self.dH = None  # J/mol, heat absorbed by beta scission
        self.MW = None  # kg/mol, molecular weight of POM
        self.gamma = None
        self.lh = None  # J/kg, latent heat of POM melting
        self.T_melt = None  # K, POM melting point
        self.slope_Tb = None  # K/s, heating rate
        self.impurity = None  # impurity percentage
        self.boundary_condition = None  # 'top q0', 'TGA'
        self.N = None  # polymer polymerization degree, MW/MW0
        self.D = None  # m2/s, polymer diffusion coefficient
        self.phase_equilibrium = None  # whether consider phase equilibrium
        self.is_H_vap = True  # whether consider enthalpy of evaporation
        self.min_interval = None  # s, tqdm output separation time
        self.check_point_step = None  # save every this number iteration

        self.L = None  # m
        self.t_end = None  # s
        self.Nx = None
        self.t_num = None
        self.t_store = None
        self.x_arr = None
        self.t_arr = None
        self.t_arg_arr = None
        self.t_store_arr = None
        self.Nt = None
        self.dt = None
        self.dx = None
        self.cfl = None  # pseudo CFL number, alpha * dt / dx ** 2

        # Evaporation related
        self.db_path = None
        self.sp_name_list = None  # [Ns,], decomposition product name list
        self.Ns = None  # number of decomposition products
        self.x_reaction = None  # [Ns,], function of temperature, polymer decomposition product mole fraction array directly from reaction
        self.x_reaction_str = None  # string of x_reaction to store in case_dict
        # self.m_polymer_init = None  # kg, initial mass of polymer
        self.eta = None  # m2/kg, effective surface area coefficient
        # self.S = None  # m2, effective surface area
        self.P = None  # Pa, ambient pressure
        self.lumped_A = None  # 1/s, lumped pre-exponential factor for polymer decomposition
        self.lumped_Ea = None  # J/mol, lumped activation energy for polymer decomposition
        self.t_end = None  # s, simulation time
        self.t_num = None  # number of sample time
        self.temp_control = None  # K, function of time, controlled temperature profile
        self.n_threshold = None  # n smaller than this will be considered as 0
        self.diffusion_coefficient = None

        self.evaporation_rate = None  # mol/m3/s [Ns,], evaporation rate for each species at current time

        self.T_mat = None  # K [Nt,Nx], temperature profile time history
        self.T_arr = None  # K [Nx,], temperature profile at current time
        self.h_mat = None  # J/kg [Nt,Nx], specific enthalpy profile time history
        self.h_arr = None  # J/kg [Nx,], specific enthalpy profile at current time
        self.phase_mat = None  # [Nt,Nx], phase profile time history
        self.phase_arr = None  # [Nx,], phase profile at current time
        self.dL_arr = None  # m [Nt,], regressed length time history
        self.dL = None  # m, regressed length at current time
        self.fp_mat = None  # mol/m3 [Nt,Nx], polymer concentration profile time history
        self.fp_arr = None  # mol/m3 [Nx,], polymer concentration profile at current time
        self.f_ten = None  # mol/m3 [Nt,Ns,Nx], products concentration profiles time history
        self.f_mat = None  # mol/m3 [Ns,Nx], products concentration profiles at current time
        # self.h_mat = None  # m [Nt,Nx], grid size
        # self.h_arr = None  # [Nx,], grid size at current time
        self.Ei_mat = None  # mol/m3/s, [Ns,Nx] evaporation rate of each product
        self.Ei_ten = None  # mol/m3/s, [Nt,Ns,Nx] evaporation rate time history
        # self.P_sat_mat = None  # mol/m3/s, [Ns,Nx] saturation pressure of each product
        # self.P_sat_ten = None  # mol/m3/s, [Nt,Ns,Nx] saturation pressure time history
        # self.x_mat = None  # mol/m3/s, [Ns,Nx] mole fraction of each product
        # self.x_ten = None  # mol/m3/s, [Nt,Ns,Nx] mole fraction time history
        # self.fp_mat_bs = None  # mol/m3 [Nt,Nx], polymer concentration profile time history before shrink
        # self.fp_arr_bs = None  # mol/m3 [Nx,], polymer concentration profile at current time before shrink
        # self.f_ten_bs = None  # mol/m3 [Nt,Ns,Nx], products concentration profiles time history before shrink
        # self.f_mat_bs = None  # mol/m3 [Ns,Nx], products concentration profiles at current time before shrink

        self.db = None  # database recording species properties
        self.df_dict = None  # df_dict generated from db
        self.MW_arr = None  # kg/mol [Ns,] molecular weight array of each product
        self.D_arr = None  # m2/s [Ns,] molecular weight array of each product
        self.H_vap = None  # J/mol [Ns,] molecular weight array of each product
        self.P_sat = None  # Pa, [Ns,] vapor pressure array of each product
        # self.Ei = None  # mol/m3/s, [Ns,Nx_liq] evaporation rate of each product
        # self.x = None  # [Ns,Nx_liq] mole fraction of each product

        self.check_point_path = None
        self.check_point_ind = None
        self.check_point_ti = None
        self.store_arr = None

    def save_case_dict(self):
        case_dict = {}
        for name in self.save_list:
            case_dict[name] = getattr(self, name)
        json.dump(case_dict, open("{}/case_dict.json".format(self.folder), "w"), indent=4, default=json_convert)
        for name in self.save_npy_list:
            if getattr(self, name) is not None:
                np.save("{}/{}.npy".format(self.folder, name), getattr(self, name))

    def save_result(self):
        for name in self.result_list:
            cur_list = []
            for ind in range(self.check_point_ind + 1):
                cur_list.append(np.load("{}/check_point/ind={}/{}.npy".format(self.folder, ind, name)))
            np.save("{}/{}.npy".format(self.folder, name), np.concatenate(cur_list))

    def save_check_point(self):
        check_point_dict = {}
        for name in self.check_point_list:
            check_point_dict[name] = getattr(self, name)
        json.dump(check_point_dict, open(self.check_point_path, "w"), indent=4, default=json_convert)
        check_point_folder = "{}/check_point/ind={}".format(self.folder, self.check_point_ind)
        if not os.path.isdir(check_point_folder):
            os.makedirs(check_point_folder)
        for name in self.result_list:
            if getattr(self, name) is not None:
                np.save("{}/{}.npy".format(check_point_folder, name), getattr(self, name))

    def load_check_point(self):
        check_point_dict = json.load(open(self.check_point_path, "r"), object_hook=json_deconvert)
        for name in check_point_dict:
            setattr(self, name, check_point_dict[name])
        for result_name, cur_name in zip(self.result_list, self.cur_list):
            result_file = "{}/check_point/ind={}/{}.npy".format(self.folder, self.check_point_ind, result_name)
            if os.path.isfile(result_file):
                cur_var = np.load(result_file).tolist()[-1]
                if hasattr(cur_var, "__len__"):
                    cur_var = np.array(cur_var)
                setattr(self, cur_name, cur_var)

    def log_print(self, *arg, **kwargs):
        print(*arg, **kwargs)
        with open(self.log_file, "a") as f:
            print(*arg, **kwargs, file=f)

    def initialize(self):
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        self.log_file = "{}/log.txt".format(self.folder)
        self.log_print("Output to {}.".format(self.folder), flush=True)
        self.check_point_path = "{}/check_point_dict.json".format(self.folder)

        # Property calculation
        # self.S = self.eta * self.m_polymer_init
        self.MW = self.N * self.MW0
        if self.lumped_A is None:
            self.lumped_A = self.A_beta * self.gamma
        if self.lumped_Ea is None:
            self.lumped_Ea = self.Ea

        # Grid definition
        self.x_arr = np.linspace(0, self.L, self.Nx)
        self.t_arr = np.linspace(0, self.t_end, self.t_num)
        self.t_arg_arr = np.arange(0, self.t_num, self.t_store, dtype=int)
        self.Nt = len(self.t_arg_arr)
        self.t_store_arr = self.t_arr[self.t_arg_arr]
        self.dt = self.t_arr[1] - self.t_arr[0]
        self.dx = self.x_arr[1] - self.x_arr[0]
        self.cfl = self.k_l / (self.rho_l * self.cp) * self.dt / self.dx**2
        self.log_print("alpha * dt / dx^2 = {}".format(self.cfl), flush=True)

        # decomposition product information loading
        if self.db is None:
            self.db = pd.read_excel(self.db_path)
        self.df_dict = {df["Name"]: df for _, df in self.db.iterrows()}
        self.MW_arr = np.array([self.df_dict[sp]["MW"] for sp in self.sp_name_list])
        self.D_arr = np.array([self.df_dict[sp]["D"] for sp in self.sp_name_list])
        self.Ns = len(self.sp_name_list)

        # stored variable setting
        for name in self.result_list:
            setattr(self, name, [])
        # self.T_mat = []
        # self.phase_mat = []
        # self.dL_arr = []
        # self.fp_mat = []
        # self.f_ten = []
        # self.Ei_ten = []

        if not os.path.isfile(self.check_point_path):
            self.save_case_dict()
            self.T_arr = self.T0 * np.ones(self.Nx)
            if self.T0 < self.T_melt:
                self.h_arr = self.cp * self.T_arr
                self.phase_arr = np.zeros(self.Nx)  # 0: solid; 0-1: s-l mixture, liquid fraction; 1: liquid; 2: gas.
            else:
                self.h_arr = self.cp * (self.T_arr - self.T_melt) + self.lh + self.cp * self.T_melt
                self.phase_arr = np.ones(self.Nx)
            self.dL = 0
            self.fp_arr = np.full(self.Nx, np.nan)
            self.f_mat = np.full((self.Ns, self.Nx), np.nan)
            self.Ei_mat = np.full((self.Ns, self.Nx), np.nan)
            self.record_state()
            self.check_point_ind = -1
            self.check_point_ti = 0
            self.store_arr = np.zeros(self.Nx)  # energy in dT stored at phase change from solid to liquid
        else:
            self.load_check_point()

    def record_state(self):
        for result_name, cur_name in zip(self.result_list, self.cur_list):
            app = getattr(self, cur_name)
            if hasattr(getattr(self, cur_name), "tolist"):
                app = app.tolist()
            setattr(self, result_name, getattr(self, result_name) + [app])
        # self.T_mat.append(self.T_arr.tolist())
        # self.phase_mat.append(self.phase_arr.tolist())
        # self.dL_arr.append(self.dL)
        # self.fp_mat.append(self.fp_arr.tolist())
        # self.f_ten.append(self.f_mat.tolist())
        # self.Ei_ten.append(self.Ei_mat.tolist())

    def get_P_sat(self, T):
        Ps = []
        for sp_name in self.sp_name_list:
            df = self.df_dict[sp_name]
            dB = df["dB"]
            Tb = df["bp"]
            rhs = (4.1012 + dB) * (T / Tb - 1) / (T / Tb - 1 / 8)
            Ps.append(10**rhs * cst.atm)
        Ps = np.array(Ps)
        return Ps

    def get_H_vap(self, T):
        if not self.is_H_vap:
            return np.zeros_like(T)

        H_vap = []
        # T = 600 * np.ones_like(T)
        for sp_name in self.sp_name_list:
            df = self.df_dict[sp_name]
            dB = df["dB"]
            Tb = df["bp"]
            drhs_dt = 56 * (4.1012 + dB) * Tb / (8 * T - Tb) ** 2
            H_vap.append(np.log(10) * drhs_dt * cst.gas_constant * T**2)
        H_vap = np.array(H_vap)
        return H_vap

    def get_evaporation_rate(self, T, f_prod, f_polymer):
        total_f = np.sum(f_prod, axis=0) + f_polymer
        x = np.zeros((self.Ns, len(T)))
        for i in range(len(T)):
            if total_f[i] > 0:
                x[:, i] = f_prod[:, i] / total_f[i]
        self.P_sat = self.get_P_sat(T)
        # self.x = x.copy()
        return x * self.P_sat * self.eta * self.rho_l * np.sqrt(1 / (2 * np.pi * cst.gas_constant * T * self.MW_arr.reshape(-1, 1)))

    def main(self):
        self.initialize()

        def rho_func(frac):
            if frac == 0:
                return self.rho_s
            elif frac == 1:
                return self.rho_l
            elif frac == 2:
                return np.nan
            else:
                return frac * self.rho_l + (1 - frac) * self.rho_s

        def k_func(frac):
            if frac == 0:
                return self.k_s
            elif frac == 1:
                return self.k_l
            elif frac == 2:
                return np.nan
            else:
                return frac * self.k_l + (1 - frac) * self.k_s

        for ti in tqdm(range(self.check_point_ti + 1, self.t_num), mininterval=self.min_interval):
            if self.q0_func is not None:
                self.q0 = self.q0_func(self.t_arr[ti])
            T_new = np.full(self.Nx, np.nan)
            h_new = np.full(self.Nx, np.nan)
            fp_new = np.full(self.Nx, np.nan)
            f_new = np.full((self.Ns, self.Nx), np.nan)
            fe_new = np.full((self.Ns + 1, self.Nx), np.nan)

            rho_arr = np.array([rho_func(p) for p in self.phase_arr])
            k_arr = np.array([k_func(p) for p in self.phase_arr])

            liquid_ind = np.where(self.phase_arr == 1)[0]
            not_gas_ind = np.where(self.phase_arr != 2)[0]
            if len(not_gas_ind) < 3:
                self.log_print("All gas!")
                break
            top_ind = not_gas_ind[0]
            inner_ind = liquid_ind[1:-1]
            boundary_ind = liquid_ind[[0, -1]] if len(liquid_ind) > 1 else liquid_ind
            ls_inner_ind = not_gas_ind[1:-1]

            self.Ei_mat = np.full((self.Ns, self.Nx), np.nan)
            self.Ei_mat[:, liquid_ind] = 0
            impurity_factor = np.maximum(1 - self.impurity * self.L / (self.L - self.dL), 0) if self.impurity is not None else 1.0
            # No liquid layer if len(liquid_ind) =< 3
            if len(liquid_ind) > 0:
                # Species equation (solved before T equation)
                for ind in liquid_ind:
                    # when just turning into liquid (only solid at the beginning)
                    # then it is initialized as only polymer, no decomposition products
                    # fp is polymer concentration [mol/m3], f_mat matrix for decomposition products
                    if np.isnan(self.fp_arr[ind]):
                        self.fp_arr[ind] = rho_arr[ind] / self.MW * self.dx
                        self.f_mat[:, ind] = 0
                if self.phase_equilibrium:
                    self.Ei_mat[:, liquid_ind] = self.get_evaporation_rate(self.T_arr[liquid_ind], self.f_mat[:, liquid_ind], self.fp_arr[liquid_ind])
                else:
                    kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.T_arr[liquid_ind]))
                    alpha_i = np.array([self.x_reaction(temp) for temp in self.T_arr[liquid_ind]]).T
                    self.Ei_mat[:, liquid_ind] = alpha_i * kr * self.fp_arr[liquid_ind] / self.dx
                self.Ei_mat[:, liquid_ind] *= impurity_factor

            # Energy equation
            def h_rate(t, h_in):
                solid_arg = np.where(h_in < self.cp * self.T_melt)[0]
                sl_arg = np.where((h_in >= self.cp * self.T_melt) & (h_in <= self.cp * self.T_melt + self.lh))[0]
                liquid_arg = np.where(h_in > self.cp * self.T_melt + self.lh)[0]
                T_in = np.zeros_like(h_in)
                T_in[solid_arg] = h_in[solid_arg] / self.cp
                T_in[sl_arg] = self.T_melt
                T_in[liquid_arg] = self.T_melt + (h_in[liquid_arg] - self.cp * self.T_melt - self.lh) / self.cp

                used_T = np.concatenate([[self.T_arr[top_ind]], T_in, [self.T_arr[-1]]])

                def get_tmp_fp_Ei(ind_list):
                    Ei_mat = np.full((self.Ns, self.Nx), np.nan)
                    Ei_mat[:, ind_list] = 0
                    fp_arr = self.fp_arr.copy()
                    f_mat = self.f_mat.copy()
                    # No liquid layer if len(liquid_ind) =< 3
                    if len(ind_list) > 0:
                        # Species equation (solved before T equation)
                        for ind in ind_list:
                            # when just turning into liquid (only solid at the beginning)
                            # then it is initialized as only polymer, no decomposition products
                            # fp is polymer concentration [mol/m3], f_mat matrix for decomposition products
                            if np.isnan(fp_arr[ind]):
                                fp_arr[ind] = rho_arr[ind] / self.MW * self.dx
                                f_mat[:, ind] = 0
                        if self.phase_equilibrium:
                            Ei_mat[:, ind_list] = self.get_evaporation_rate(self.T_arr[ind_list], f_mat[:, ind_list], fp_arr[ind_list])
                        else:
                            kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.T_arr[ind_list]))
                            alpha_i = np.array([self.x_reaction(temp) for temp in self.T_arr[ind_list]]).T
                            Ei_mat[:, ind_list] = alpha_i * kr * fp_arr[ind_list] / self.dx
                        Ei_mat[:, ind_list] *= impurity_factor
                    return fp_arr, Ei_mat

                fp_arr, Ei_mat = get_tmp_fp_Ei(ls_inner_ind[liquid_arg])
                Q_rxn = np.zeros_like(h_in)
                kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * T_in[liquid_arg])) * impurity_factor
                Q_rxn[liquid_arg] = self.dH * kr * fp_arr[ls_inner_ind[liquid_arg]] / self.dx

                self.H_vap = self.get_H_vap(T_in[liquid_arg])
                evap_heat = np.zeros_like(h_in)
                evap_heat[liquid_arg] = np.sum(Ei_mat[:, ls_inner_ind[liquid_arg]] * self.H_vap, axis=0)

                tmp_arr = (k_arr[ls_inner_ind] / self.dx**2 * (used_T[2:] - 2 * T_in + used_T[:-2]) - Q_rxn - evap_heat) / rho_arr[ls_inner_ind]
                return tmp_arr

            def h_top_rate(t, h_top):
                k_avg = (k_arr[top_ind] + k_arr[top_ind + 1]) / 2
                tmp_arr = (k_avg * (self.T_arr[top_ind + 1] - self.T_arr[top_ind]) / self.dx + self.q0) / self.dx / rho_arr[top_ind]
                return tmp_arr

            h_new[ls_inner_ind] = advance_rk4(h_rate, ti * self.dt, self.h_arr[ls_inner_ind], self.dt)
            if self.boundary_condition == "top q0":
                h_new[top_ind] = advance_rk4(h_top_rate, ti * self.dt, self.h_arr[top_ind], self.dt)
                h_new[-1] = h_new[-2]
            elif self.boundary_condition == "TGA":
                Tb = self.T0 + self.slope_Tb * ti * self.dt
                if Tb < self.T_melt:
                    hb = self.cp * Tb
                else:
                    hb = self.cp * (Tb - self.T_melt) + self.lh + self.cp * self.T_melt
                # h_new[top_ind] = h_new[top_ind + 1]
                h_new[top_ind] = hb
                h_new[-1] = hb
            else:
                raise Exception("Unknown boundary condition: {}!".format(self.boundary_condition))
            solid_arg = np.where(h_new < self.cp * self.T_melt)[0]
            sl_arg = np.where((h_new >= self.cp * self.T_melt) & (h_new <= self.cp * self.T_melt + self.lh))[0]
            liquid_arg = np.where(h_new > self.cp * self.T_melt + self.lh)[0]
            T_new[solid_arg] = h_new[solid_arg] / self.cp
            T_new[sl_arg] = self.T_melt
            T_new[liquid_arg] = self.T_melt + (h_new[liquid_arg] - self.cp * self.T_melt - self.lh) / self.cp
            self.phase_arr[solid_arg] = 0
            self.phase_arr[sl_arg] = (h_new[sl_arg] - self.cp * self.T_melt) / self.lh
            self.phase_arr[liquid_arg] = 1

            if len(liquid_ind) > 0:
                # fe mean "phi entire". Phi=letter for concentration, entire means combined polymer and decomposition products
                def fe_rate(t, fe_in):
                    # only inner points are updated, top and bottom are like boundary conditions assigned
                    # bottom (solid-liquid interphase is always pure polymer)
                    fe_top = np.concatenate([[self.fp_arr[liquid_ind[0]]], self.f_mat[:, liquid_ind[0]]])
                    fe_bottom = np.concatenate([[self.fp_arr[liquid_ind[-1]]], self.f_mat[:, liquid_ind[-1]]])
                    # to define all the grid points that are liquid
                    used_fe = np.hstack([fe_top.reshape(-1, 1), fe_in, fe_bottom.reshape(-1, 1)])

                    # terms within the species equation
                    kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.T_arr[inner_ind]))
                    rxn_rate = kr * fe_in[0, :] * 2 / self.N
                    alpha_i = np.array([self.x_reaction(temp) for temp in self.T_arr[inner_ind]]).T
                    beta_i = self.N * alpha_i / sum([(ii + 1) * alpha_i[ii, :] for ii in range(alpha_i.shape[0])])
                    Ei = self.get_evaporation_rate(self.T_arr[inner_ind], fe_in[1:, :], fe_in[0, :])
                    # for decomposition products (reaction + evaporation, Ei)
                    source_i = beta_i * rxn_rate - Ei * self.dx
                    # then combined (stack) polymer (first element with no evaporation) and decomposition products
                    source = np.vstack([-rxn_rate.reshape(1, -1), source_i])
                    tmp_arr = np.concatenate([[self.D], self.D_arr]).reshape(-1, 1) / self.dx**2 * (used_fe[:, 2:] - 2 * fe_in + used_fe[:, :-2]) + source
                    return tmp_arr

                def fe_boundary_rate(t, fe_bd):
                    # terms within the species equation
                    kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.T_arr[boundary_ind]))
                    rxn_rate = kr * fe_bd[0, :] * 2 / self.N
                    alpha_i = np.array([self.x_reaction(temp) for temp in self.T_arr[boundary_ind]]).T
                    beta_i = self.N * alpha_i / sum([(ii + 1) * alpha_i[ii, :] for ii in range(alpha_i.shape[0])])
                    Ei = self.get_evaporation_rate(self.T_arr[boundary_ind], fe_bd[1:, :], fe_bd[0, :])
                    # for decomposition products (reaction + evaporation, Ei)
                    source_i = beta_i * rxn_rate - Ei * self.dx
                    # then combined (stack) polymer (first element with no evaporation) and decomposition products
                    source = np.vstack([-rxn_rate.reshape(1, -1), source_i])
                    return source

                def fe_liquid_rate(t, fe_lq):
                    # terms within the species equation
                    kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.T_arr[liquid_ind]))
                    alpha_i = np.array([self.x_reaction(temp) for temp in self.T_arr[liquid_ind]]).T
                    if self.phase_equilibrium:
                        Ei = self.get_evaporation_rate(self.T_arr[liquid_ind], fe_lq[1:, :], fe_lq[0, :])
                    else:
                        Ei = alpha_i * kr * fe_lq[0, :] / self.dx
                    Ei *= impurity_factor
                    # for decomposition products (reaction + evaporation, Ei)
                    source_i = alpha_i * kr * fe_lq[0, :] - Ei * self.dx
                    source_p = sum([(ii + 1) * alpha_i[ii, :] for ii in range(alpha_i.shape[0])]) / self.N * kr * fe_lq[0, :]
                    # then combined (stack) polymer (first element with no evaporation) and decomposition products
                    source = np.vstack([-source_p.reshape(1, -1), source_i])
                    return source

                fe_mat = np.vstack([self.fp_arr.reshape(1, -1), self.f_mat])
                fe_new[:, liquid_ind] = advance_rk4(fe_liquid_rate, ti * self.dt, fe_mat[:, liquid_ind], self.dt)

                # if len(inner_ind) > 0:
                #     fe_new[:, inner_ind] = advance_rk4(fe_rate, ti * self.dt, fe_mat[:, inner_ind], self.dt)
                # fe_new[:, boundary_ind] = advance_rk4(fe_boundary_rate, ti * self.dt, fe_mat[:, boundary_ind], self.dt)

                # if based on a too large time step some species concentration becomes negative.
                # In this case it is forced to 0. To be fixed with non-fixed time-step
                fe_new[fe_new < 0] = 0
                fp_new = fe_new[0, :]
                f_new = fe_new[1:, :]

                h_arr = fp_new[liquid_ind] * self.MW / rho_arr[liquid_ind] + np.sum(f_new[:, liquid_ind] * self.MW_arr.reshape(-1, 1) / rho_arr[liquid_ind], axis=0)
                h_arr = h_arr[::-1]

                # thickness of the gas-phase (from balancing gas-phase mass produced and liquid-phase mass consumed)
                self.dL += np.sum(self.Ei_mat[:, liquid_ind] * self.dt * self.MW_arr.reshape(-1, 1) / self.rho_l * self.dx)

                sum_h = np.array(list(itertools.accumulate(h_arr))) - h_arr / 2
                new_top_ind = int(np.round(self.dL / self.dx))

                def get_new_f(f):
                    f_in = f.copy()
                    f_arr = f_in[liquid_ind][::-1]
                    c_arr = f_arr / h_arr
                    c_fn = scipy.interpolate.interp1d(sum_h, c_arr, bounds_error=False, fill_value=(c_arr[0], c_arr[-1]))
                    new_c = c_fn(self.dx * np.arange(len(sum_h)) + self.dx / 2)
                    new_f = new_c * self.dx
                    new_f = new_f[::-1]
                    f_in[liquid_ind] = new_f
                    f_in[top_ind:new_top_ind] = np.nan
                    return f_in

                fp_new = get_new_f(fp_new)
                for i in range(f_new.shape[0]):
                    f_new[i, :] = get_new_f(f_new[i, :])
                T_new[top_ind:new_top_ind] = np.nan

            self.phase_arr[top_ind : int(np.round(self.dL / self.dx))] = 2
            self.T_arr = T_new.copy()
            self.h_arr = h_new.copy()
            self.fp_arr = fp_new.copy()
            self.f_mat = f_new.copy()
            # self.Ei_mat = np.full((self.Ns, self.Nx), np.nan)
            # self.Ei_mat[:, liquid_ind] = self.Ei.copy()

            if ti in self.t_arg_arr:
                self.record_state()
                if ti % self.check_point_step == 0:
                    self.check_point_ind += 1
                    self.check_point_ti = ti
                    self.save_check_point()
                    self.initialize()

        if len(self.T_mat) > 0:
            self.check_point_ind += 1
            self.check_point_ti = ti
            self.save_check_point()
        self.save_result()


def case42():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case42".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer"]
    # tt.sp_name_list = ["Styrene"]

    ### SF-HyChem setting
    # tt.x_reaction = lambda T: np.array([8, 4, 2, 1, 1], dtype=float)
    # tt.x_reaction_str = 'lambda T: np.array([8, 4, 2, 1, 1], dtype=float)'
    tt.x_reaction = lambda T: np.array([0.9, 0.05, 0.05], dtype=float)
    tt.x_reaction_str = "lambda T: np.array([0.9, 0.05, 0.05], dtype=float)"
    # tt.x_reaction = lambda T: np.array([1 - 33288 * T ** -2.174 - 714581 * T ** -2.743, 33288 * T ** -2.174, 714581 * T ** -2.743])  # expt.
    # tt.x_reaction_str = 'lambda T: np.array([1 - 33288 * T ** -2.174 - 714581 * T ** -2.743, 33288 * T ** -2.174, 714581 * T ** -2.743])  # expt.'
    # tt.x_reaction = lambda T: np.array([1 - 487185 * T ** -2.413 - 6e9 * T ** -4.192, 487185 * T ** -2.413, 6e9 * T ** -4.192])  # CRECK model
    # tt.x_reaction_str = 'lambda T: np.array([1 - 487185 * T ** -2.413 - 6e9 * T ** -4.192, 487185 * T ** -2.413, 6e9 * T ** -4.192])  # CRECK model'
    # tt.x_reaction = lambda T: np.array([1.0])
    # tt.x_reaction_str = 'lambda T: np.array([1.0])'

    tt.T0 = 300  # K, initial temperature
    tt.q0 = 1e5  # W/m2, heat flux from gas phase
    # tt.m_polymer_init = 10e-3  # kg, initial mass of polymer
    tt.eta = 7.4e-6  # m2/kg, effective surface area coefficient
    tt.boundary_condition = "top q0"

    ### physical properties setting
    # tt.lumped_A = 2e11  # 1/s, lumped pre-exponential factor for polymer decomposition
    # tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    # tt.T_melt = 165 + 273  # K, melting temperature
    # tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    # tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    # tt.rho_s = 1.42e3  # kg/m3, solid phase density
    # tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    # tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    # tt.cv = 35 / tt.MW0  # J/kg·K
    # tt.cp = tt.cv
    # tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    # tt.lh = 150e3  # J/kg, latent heat of fuel melting
    # tt.N = 3000  # number of polymer degree
    # tt.D = 0  # m2/s, polymer diffusion coefficient

    tt.lumped_A = 2e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 240 + 273  # K, melting temperature
    tt.k_s = 0.16  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.135  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.05e3  # kg/m3, solid phase density
    tt.rho_l = 0.975e3  # kg/m3, liquid phase density
    tt.MW0 = 104e-3  # kg/mol, C8H8 (styrene) molecular weight
    tt.cv = 1.3e3  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 73.4e3  # J/mol, heat absorbed by beta scission
    tt.lh = 45e3  # J/kg, latent heat of fuel melting
    tt.N = 3000  # number of polymer degree
    tt.D = 3e-12  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 500 + 1
    tt.t_num = 100000 + 1
    tt.t_store = 100

    tt.phase_equilibrium = False
    tt.min_interval = 2
    tt.check_point_step = 10000
    # tt.main()
    return tt


def case45():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case45".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 300  # K, initial temperature
    tt.q0 = 1e5  # W/m2, heat flux from gas phase
    tt.boundary_condition = "top q0"

    ### physical properties setting
    tt.lumped_A = 2 * 1.8e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 30e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 165 + 273  # K, melting temperature
    tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.42e3  # kg/m3, solid phase density
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.lh = 150e3  # J/kg, latent heat of fuel melting
    tt.N = 3333  # number of polymer degree
    tt.D = 0  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 500 + 1
    tt.t_num = 100000 + 1
    tt.t_store = 100

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 10000
    tt.main()
    return tt


def case46():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case46".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 300  # K, initial temperature
    tt.q0 = 1e5  # W/m2, heat flux from gas phase

    ### physical properties setting
    tt.lumped_A = 2 * 1.8e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 30e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 165 + 273  # K, melting temperature
    tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.42e3  # kg/m3, solid phase density
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.lh = 150e3  # J/kg, latent heat of fuel melting
    tt.N = 3333  # number of polymer degree
    tt.D = 0  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 5000 + 1
    tt.t_num = 10000000 + 1
    tt.t_store = 1000

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 120
    tt.check_point_step = 100000
    tt.main()


def case47():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case47".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 300  # K, initial temperature
    tt.q0 = 1e5  # W/m2, heat flux from gas phase
    tt.boundary_condition = "top q0"

    ### physical properties setting
    tt.lumped_A = 2 * 1.8e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 30e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 165 + 273  # K, melting temperature
    tt.k_s = 0.14  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.2e3  # kg/m3, solid phase density
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.lh = 150e3  # J/kg, latent heat of fuel melting
    tt.N = 3333  # number of polymer degree
    tt.D = 0  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 4e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 10000 + 1
    tt.t_num = 1000000 + 1
    tt.t_store = 1000

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 10000
    tt.main()
    return tt


def case52():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case52".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 100 + 273  # K, initial temperature
    tt.slope_Tb = 5 / 60  # K/s, heating rate
    tt.impurity = 0.03  # polymer impurity percentage
    tt.boundary_condition = "TGA"

    ### physical properties setting
    tt.lumped_A = 2 * 1.8e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 30e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 165 + 273  # K, melting temperature
    tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.42e3  # kg/m3, solid phase density
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.lh = 150e3  # J/kg, latent heat of fuel melting
    tt.N = 3333  # number of polymer degree
    tt.D = 0  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    tt.L = 4e-2  # m
    tt.t_end = 4800  # s
    tt.Nx = 200 + 1
    tt.t_num = 100000 + 1
    tt.t_store = 100

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 10000
    tt.main()
    return tt


def case53():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case53".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1 - 33288 * T**-2.174 - 714581 * T**-2.743, 33288 * T**-2.174, 714581 * T**-2.743])  # expt.
    tt.x_reaction_str = "lambda T: np.array([1 - 33288 * T ** -2.174 - 714581 * T ** -2.743, 33288 * T ** -2.174, 714581 * T ** -2.743])  # expt."

    tt.T0 = 300 + 273  # K, initial temperature
    tt.slope_Tb = 10 / 60  # K/s, heating rate
    tt.impurity = 0.03  # polymer impurity percentage
    tt.boundary_condition = "TGA"

    ### physical properties setting
    tt.lumped_A = 2e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 240 + 273  # K, melting temperature
    tt.k_s = 0.16  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.135  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.05e3  # kg/m3, solid phase density
    tt.rho_l = 0.975e3  # kg/m3, liquid phase density
    tt.MW0 = 104e-3  # kg/mol, C8H8 (styrene) molecular weight
    tt.cv = 1.3e3  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 73.4e3  # J/mol, heat absorbed by beta scission
    tt.lh = 45e3  # J/kg, latent heat of fuel melting
    tt.N = 3000  # number of polymer degree
    tt.D = 3e-12  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    tt.L = 4e-2  # m
    tt.t_end = 1200  # s
    tt.Nx = 200 + 1
    tt.t_num = 100000 + 1
    tt.t_store = 100

    tt.phase_equilibrium = False
    tt.is_H_vap = True
    tt.min_interval = 2
    tt.check_point_step = 10000
    tt.main()
    return tt


def case54():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case54".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1 - 33288 * T**-2.174 - 714581 * T**-2.743, 33288 * T**-2.174, 714581 * T**-2.743])  # expt.
    tt.x_reaction_str = "lambda T: np.array([1 - 33288 * T ** -2.174 - 714581 * T ** -2.743, 33288 * T ** -2.174, 714581 * T ** -2.743])  # expt."

    tt.T0 = 300 + 273  # K, initial temperature
    tt.slope_Tb = 10 / 60  # K/s, heating rate
    tt.impurity = 0.03  # polymer impurity percentage
    tt.eta = 7.4e-6  # m2/kg, effective surface area coefficient
    tt.boundary_condition = "TGA"

    ### physical properties setting
    tt.lumped_A = 2e11 * 3000  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 240 + 273  # K, melting temperature
    tt.k_s = 0.16  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.135  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.05e3  # kg/m3, solid phase density
    tt.rho_l = 0.975e3  # kg/m3, liquid phase density
    tt.MW0 = 104e-3  # kg/mol, C8H8 (styrene) molecular weight
    tt.cv = 1.3e3  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 73.4e3  # J/mol, heat absorbed by beta scission
    tt.lh = 45e3  # J/kg, latent heat of fuel melting
    tt.N = 3000  # number of polymer degree
    tt.D = 3e-12  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    tt.L = 4e-2  # m
    tt.t_end = 1200  # s
    tt.Nx = 200 + 1
    tt.t_num = 100000 + 1
    tt.t_store = 100

    tt.phase_equilibrium = True
    tt.is_H_vap = True
    tt.min_interval = 2
    tt.check_point_step = 10000
    tt.main()
    return tt


def case55():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case55".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1 - 33288 * T**-2.174 - 714581 * T**-2.743, 33288 * T**-2.174, 714581 * T**-2.743])  # expt.
    tt.x_reaction_str = "lambda T: np.array([1 - 33288 * T ** -2.174 - 714581 * T ** -2.743, 33288 * T ** -2.174, 714581 * T ** -2.743])  # expt."

    tt.T0 = 300 + 273  # K, initial temperature
    tt.slope_Tb = 10 / 60  # K/s, heating rate
    tt.impurity = 0.03  # polymer impurity percentage
    tt.eta = 7.4e-6  # m2/kg, effective surface area coefficient
    tt.boundary_condition = "TGA"

    ### physical properties setting
    tt.lumped_A = 2e11 * 3000  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 240 + 273  # K, melting temperature
    tt.k_s = 0.16  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.135  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.05e3  # kg/m3, solid phase density
    tt.rho_l = 0.975e3  # kg/m3, liquid phase density
    tt.MW0 = 104e-3  # kg/mol, C8H8 (styrene) molecular weight
    tt.cv = 1.3e3  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 73.4e3  # J/mol, heat absorbed by beta scission
    tt.lh = 45e3  # J/kg, latent heat of fuel melting
    tt.N = 3000  # number of polymer degree
    tt.D = 3e-12  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    tt.L = 1e-2  # m
    tt.t_end = 1200  # s
    tt.Nx = 200 + 1
    tt.t_num = 100000 + 1
    tt.t_store = 100

    tt.phase_equilibrium = True
    tt.is_H_vap = True
    tt.min_interval = 2
    tt.check_point_step = 10000
    tt.main()
    return tt


def case56():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case56".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer"]
    # tt.sp_name_list = ["Styrene"]

    ### SF-HyChem setting
    # tt.x_reaction = lambda T: np.array([8, 4, 2, 1, 1], dtype=float)
    # tt.x_reaction_str = 'lambda T: np.array([8, 4, 2, 1, 1], dtype=float)'
    # tt.x_reaction = lambda T: np.array([0.9, 0.05, 0.05], dtype=float)
    # tt.x_reaction_str = "lambda T: np.array([0.9, 0.05, 0.05], dtype=float)"
    tt.x_reaction = lambda T: np.array([1 - 33288 * T**-2.174 - 714581 * T**-2.743, 33288 * T**-2.174, 714581 * T**-2.743])  # expt.
    tt.x_reaction_str = "lambda T: np.array([1 - 33288 * T ** -2.174 - 714581 * T ** -2.743, 33288 * T ** -2.174, 714581 * T ** -2.743])  # expt."
    # tt.x_reaction = lambda T: np.array([1 - 487185 * T ** -2.413 - 6e9 * T ** -4.192, 487185 * T ** -2.413, 6e9 * T ** -4.192])  # CRECK model
    # tt.x_reaction_str = 'lambda T: np.array([1 - 487185 * T ** -2.413 - 6e9 * T ** -4.192, 487185 * T ** -2.413, 6e9 * T ** -4.192])  # CRECK model'
    # tt.x_reaction = lambda T: np.array([1.0])
    # tt.x_reaction_str = 'lambda T: np.array([1.0])'

    tt.T0 = 300  # K, initial temperature
    tt.q0 = 1e5  # W/m2, heat flux from gas phase
    # tt.m_polymer_init = 10e-3  # kg, initial mass of polymer
    tt.eta = 7.4e-6  # m2/kg, effective surface area coefficient
    tt.boundary_condition = "top q0"

    ### physical properties setting
    # tt.lumped_A = 2e11  # 1/s, lumped pre-exponential factor for polymer decomposition
    # tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    # tt.T_melt = 165 + 273  # K, melting temperature
    # tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    # tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    # tt.rho_s = 1.42e3  # kg/m3, solid phase density
    # tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    # tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    # tt.cv = 35 / tt.MW0  # J/kg·K
    # tt.cp = tt.cv
    # tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    # tt.lh = 150e3  # J/kg, latent heat of fuel melting
    # tt.N = 3000  # number of polymer degree
    # tt.D = 0  # m2/s, polymer diffusion coefficient

    tt.lumped_A = 2e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 240 + 273  # K, melting temperature
    tt.k_s = 0.16  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.135  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.05e3  # kg/m3, solid phase density
    tt.rho_l = 0.975e3  # kg/m3, liquid phase density
    tt.MW0 = 104e-3  # kg/mol, C8H8 (styrene) molecular weight
    tt.cv = 1.3e3  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 73.4e3  # J/mol, heat absorbed by beta scission
    tt.lh = 45e3  # J/kg, latent heat of fuel melting
    tt.N = 3000  # number of polymer degree
    tt.D = 3e-12  # m2/s, polymer diffusion coefficient
    ### physical properties setting

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    # tt.L = 2e-2  # m
    # tt.t_end = 400  # s
    # tt.Nx = 500 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 4e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 10000 + 1
    tt.t_num = 10000000 + 1
    tt.t_store = 1000
    ### Grid and time step setting

    tt.phase_equilibrium = False
    tt.min_interval = 2
    tt.check_point_step = 100000
    # tt.main()
    return tt


def case57():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case57".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 300  # K, initial temperature
    tt.q0_func = lambda t: 1e5 + 1e5 * np.sin(2 * cst.pi * t / (10 + 9 * np.sin(2 * cst.pi * t / 400))) * (
        (int(t / (10 + 9 * np.sin(2 * cst.pi * t / 400))) + 1) % 2
    )  # W/m2, heat flux from gas phase as a function of time
    tt.q0_func_str = "lambda t: 1e5 + 1e5 * np.sin(2 * cst.pi * t / (10 + 9 * np.sin(2 * cst.pi * t / 400))) * ((int(t / (10 + 9 * np.sin(2 * cst.pi * t / 400))) + 1) % 2)"
    tt.boundary_condition = "top q0"

    ### physical properties setting
    tt.lumped_A = 2 * 1.8e13  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 30e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.T_melt = 165 + 273  # K, melting temperature
    tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.42e3  # kg/m3, solid phase density
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.lh = 150e3  # J/kg, latent heat of fuel melting
    tt.N = 3333  # number of polymer degree
    tt.D = 0  # m2/s, polymer diffusion coefficient

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 5000 + 1
    tt.t_num = 10000000 + 1
    tt.t_store = 1000

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 100000
    tt.main()


def case58():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case58".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 300  # K, initial temperature
    tt.q0_func = lambda t: 1e5 * (1 - np.exp(-t / 20e-3)) + (1e5 * (1 - np.exp(-t / 20e-3)) + 0.5e5 * (1 - np.exp(-t / 20e-3))) * np.sin(
        2 * np.pi * t / 10e-3
    )  # W/m2, heat flux from gas phase as a function of time
    tt.q0_func_str = "lambda t: 1e5 * (1 - np.exp(-t / 20e-3)) + (1e5 * (1 - np.exp(-t / 20e-3)) + 0.5e5 * (1 - np.exp(-t / 20e-3))) * np.sin(2 * np.pi * t / 10e-3)"
    tt.boundary_condition = "top q0"

    ### physical properties setting
    tt.D = 0  # m2/s, polymer diffusion coefficient

    prop_dict = json.load(open("{}/output/property_dict/POM.json".format(work_dir), "r"))
    tt.T_melt = prop_dict["T_melt"]
    tt.k_l = prop_dict["k_l"]
    tt.k_s = prop_dict["k_s"]
    tt.rho_l = prop_dict["rho_l"]
    tt.rho_s = prop_dict["rho_s"]
    tt.MW0 = prop_dict["MW0"]
    A_beta = prop_dict["A_beta"]
    tt.lumped_Ea = prop_dict["lumped_Ea"]
    tt.dH = prop_dict["dH"]
    MW = prop_dict["MW"]
    tt.lh = prop_dict["lh"]
    tt.cp = prop_dict["cp"]

    tt.N = MW / tt.MW0  # number of polymer degree
    tt.lumped_A = 2 * A_beta  # 1/s

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 400  # s
    tt.Nx = 5000 + 1
    tt.t_num = 10000000 + 1
    tt.t_store = 1000

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 100000
    tt.main()


def case59():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case59".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.T0 = 500  # K, initial temperature
    tt.q0_func = lambda t: 1e5 * (1 - np.exp(-t / 20e-3)) + (1e5 * (1 - np.exp(-t / 20e-3)) + 0.5e5 * (1 - np.exp(-t / 20e-3))) * np.sin(
        2 * np.pi * t / 10e-3
    )  # W/m2, heat flux from gas phase as a function of time
    tt.q0_func_str = "lambda t: 1e5 * (1 - np.exp(-t / 20e-3)) + (1e5 * (1 - np.exp(-t / 20e-3)) + 0.5e5 * (1 - np.exp(-t / 20e-3))) * np.sin(2 * np.pi * t / 10e-3)"
    tt.boundary_condition = "top q0"

    ### physical properties setting
    tt.D = 0  # m2/s, polymer diffusion coefficient

    prop_dict = json.load(open("{}/output/property_dict/POM.json".format(work_dir), "r"))
    tt.T_melt = prop_dict["T_melt"]
    tt.k_l = prop_dict["k_l"]
    tt.k_s = prop_dict["k_s"]
    tt.rho_l = prop_dict["rho_l"]
    tt.rho_s = prop_dict["rho_s"]
    tt.MW0 = prop_dict["MW0"]
    A_beta = prop_dict["A_beta"]
    tt.lumped_Ea = prop_dict["lumped_Ea"]
    tt.dH = prop_dict["dH"]
    MW = prop_dict["MW"]
    tt.lh = prop_dict["lh"]
    tt.cp = prop_dict["cp"]

    tt.N = MW / tt.MW0  # number of polymer degree
    tt.lumped_A = 2 * A_beta  # 1/s

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 0.1  # s
    tt.Nx = 5000 + 1
    tt.t_num = 10000 + 1
    tt.t_store = 1

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 100
    tt.main()


def case60():
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case60".format(work_dir)
    tt.db_path = "{}/data/polymer_evaporation.xlsx".format(work_dir)
    tt.sp_name_list = ["CH2O"]

    ### SF-HyChem setting
    tt.x_reaction = lambda T: np.array([1.0])
    tt.x_reaction_str = "lambda T: np.array([1.0])"

    tt.q0_func = lambda t: 1e5 * (1 - np.exp(-t / 20e-3)) + (1e5 * (1 - np.exp(-t / 20e-3)) + 0.5e5 * (1 - np.exp(-t / 20e-3))) * np.sin(
        2 * np.pi * t / 10e-3
    )  # W/m2, heat flux from gas phase as a function of time
    tt.q0_func_str = "lambda t: 1e5 * (1 - np.exp(-t / 20e-3)) + (1e5 * (1 - np.exp(-t / 20e-3)) + 0.5e5 * (1 - np.exp(-t / 20e-3))) * np.sin(2 * np.pi * t / 10e-3)"
    tt.boundary_condition = "top q0"

    ### physical properties setting
    tt.D = 0  # m2/s, polymer diffusion coefficient

    prop_dict = json.load(open("{}/output/property_dict/POM.json".format(work_dir), "r"))
    tt.T_melt = prop_dict["T_melt"]
    tt.T0 = tt.T_melt  # K, initial temperature
    tt.k_l = prop_dict["k_l"]
    tt.k_s = prop_dict["k_s"]
    tt.rho_l = prop_dict["rho_l"]
    tt.rho_s = prop_dict["rho_s"]
    tt.MW0 = prop_dict["MW0"]
    A_beta = prop_dict["A_beta"]
    tt.lumped_Ea = prop_dict["lumped_Ea"]
    tt.dH = prop_dict["dH"]
    MW = prop_dict["MW"]
    tt.lh = prop_dict["lh"]
    tt.cp = prop_dict["cp"]

    tt.N = MW / tt.MW0  # number of polymer degree
    tt.lumped_A = 2 * A_beta  # 1/s

    ### Grid and time step setting
    # tt.L = 1e-2  # m
    # tt.t_end = 800  # s
    # tt.Nx = 50 + 1
    # tt.t_num = 100000 + 1
    # tt.t_store = 100

    tt.L = 2e-2  # m
    tt.t_end = 0.1  # s
    tt.Nx = 5000 + 1
    tt.t_num = 10000 + 1
    tt.t_store = 1

    tt.phase_equilibrium = False
    tt.is_H_vap = False
    tt.min_interval = 2
    tt.check_point_step = 1000
    tt.main()


def anchor_point():
    pass


if __name__ == "__main__":
    # tt = case42()
    # tt = case45()
    # tt = case52()
    # tt = case53()
    # tt = case54()
    # tt = case55()
    # tt = case57()
    # tt = case58()
    # tt = case59()
    tt = case60()
    pass
