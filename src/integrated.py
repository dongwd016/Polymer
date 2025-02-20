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
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["animation.ffmpeg_path"] = "C:/Program Files/ffmpeg/bin/ffmpeg.exe"
plt.rcParams["animation.embed_limit"] = 50
mu_sb = "\u03bc"
deg_sb = "\u00b0"

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


def advance_ee(dy_dt, t, y, dt):
    return y + dy_dt(y, t) * dt


def advance_rk4(dy_dt, t, y, dt):
    k1 = dt * dy_dt(t, y)
    k2 = dt * dy_dt(t + dt / 2, y + k1 / 2)
    k3 = dt * dy_dt(t + dt / 2, y + k2 / 2)
    k4 = dt * dy_dt(t + dt, y + k3)
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y + k


class Polymer:
    def __init__(self):
        self.save_list = ['folder', 'T0', 'q0', 'qb', 'k_s', 'k_l', 'rho_s', 'rho_l', 'MW0', 'cv', 'cp', 'A_beta', 'Ea', 'dH', 'MW', 'gamma', 'lh', 'T_melt', 'slope_Tb', 'L',
                          't_end', 'Nx', 't_num', 't_store', 'Nt', 'dt', 'dx', 'cfl', 'db_path', 'sp_name_list', 'Ns', 'x_reaction_str', 'm_polymer_init', 'eta', 'S', 'P',
                          'lumped_A', 'lumped_Ea', 't_end', 't_num', 'temp_control', 'n_threshold', 'diffusion_coefficient', 'N', 'D']
        self.result_list = ['x_arr', 't_arr', 't_arg_arr', 't_store_arr', 'T_mat', 'phase_mat', 'dL_arr', 'fp_mat', 'f_ten']

        self.folder = None

        # Transient related (heat transfer, non-uniform temperature)
        self.T0 = None  # K
        self.q0 = None  # W/m2, heat flux from gas phase
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
        self.N = None  # polymer polymerization degree, MW/MW0
        self.D = None  # m2/s, polymer diffusion coefficient

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
        self.m_polymer_init = None  # kg, initial mass of polymer
        self.eta = None  # m2/kg, effective surface area coefficient
        self.S = None  # m2, effective surface area
        self.P = None  # Pa, ambient pressure
        self.lumped_A = None  # 1/s, lumped pre-exponential factor for polymer decomposition
        self.lumped_Ea = None  # J/mol, lumped activation energy for polymer decomposition
        self.t_end = None  # s, simulation time
        self.t_num = None  # number of sample time
        self.temp_control = None  # K, function of time, controlled temperature profile
        self.n_threshold = None  # n smaller than this will be considered as 0
        self.diffusion_coefficient = None

        self.t = None  # s, current time
        self.evaporation_rate = None  # mol/m3/s [Ns,], evaporation rate for each species at current time

        self.T_mat = None  # K [Nt,Nx], temperature profile time history
        self.T_arr = None  # K [Nx,], temperature profile at current time
        self.phase_mat = None  # [Nt,Nx], phase profile time history
        self.phase_arr = None  # [Nx,], phase profile at current time
        self.dL_arr = None  # m [Nt,], regressed length time history
        self.dL = None  # m, regressed length at current time
        self.fp_mat = None  # mol/m3 [Nt,Nx], polymer concentration profile time history
        self.fp_arr = None  # mol/m3 [Nx,], polymer concentration profile at current time
        self.f_ten = None  # mol/m3 [Nt,Ns,Nx], products concentration profiles time history
        self.f_mat = None  # mol/m3 [Ns,Nx], products concentration profiles at current time

        self.db = None  # database recording species properties
        self.df_dict = None  # df_dict generated from db
        self.MW_arr = None  # kg/mol [Ns,] molecular weight array of each product
        self.D_arr = None  # m2/s [Ns,] molecular weight array of each product
        self.H_vap = None  # J/mol [Ns,] molecular weight array of each product
        self.P_sat = None  # Pa, [Ns,] vapor pressure array of each product
        self.Ei = None  # mol/m3/s, [Ns,Nx] evaporation rate of each product

    def save_case_dict(self):
        case_dict = {}
        for name in self.save_list:
            case_dict[name] = getattr(self, name)
        json.dump(case_dict, open("{}/case_dict.json".format(self.folder), "w"), indent=4, default=json_convert)

    def save_result(self):
        for name in self.result_list:
            if getattr(self, name) is not None:
                np.save("{}/{}.npy".format(self.folder, name), getattr(self, name))

    def initialize(self):
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        print('Output to {}.'.format(self.folder))

        # Property calculation
        self.S = self.eta * self.m_polymer_init
        self.MW = self.N * self.MW0
        if self.lumped_A is None:
            self.lumped_A = self.A_beta * self.gamma
        if self.lumped_Ea is None:
            self.lumped_Ea = self.Ea

        # Grid definition
        self.x_arr = np.linspace(0, self.L, self.Nx)
        self.t_arr = np.linspace(0, self.t_end, self.t_num)
        self.t_arg_arr = np.arange(0, self.t_num, self.t_store, dtype=int)
        self.t_store_arr = self.t_arr[self.t_arg_arr]
        self.dt = self.t_arr[1] - self.t_arr[0]
        self.dx = self.x_arr[1] - self.x_arr[0]
        self.cfl = self.k_l / (self.rho_l * self.cp) * self.dt / self.dx ** 2
        print('alpha * dt / dx^2 = {}'.format(self.cfl))

        # decomposition product information loading
        if self.db is None:
            self.db = pd.read_excel(self.db_path)
        self.df_dict = {df["Name"]: df for _, df in self.db.iterrows()}
        self.MW_arr = np.array([self.df_dict[sp]['MW'] for sp in self.sp_name_list])
        self.D_arr = np.array([self.df_dict[sp]['D'] for sp in self.sp_name_list])
        self.Ns = len(self.sp_name_list)

        # stored variable setting
        self.Nt = len(self.t_arg_arr)
        # self.T_mat = np.full((self.Nt, self.Nx), np.nan)
        # self.phase_mat = np.full((self.Nt, self.Nx), np.nan)
        # self.dL_arr = np.full(self.Nt, np.nan)
        # self.fp_mat = np.full((self.Nt, self.Nx), np.nan)
        # self.f_ten = np.full((self.Nt, self.Ns, self.Nx), np.nan)
        self.T_mat = []
        self.phase_mat = []
        self.dL_arr = []
        self.fp_mat = []
        self.f_ten = []

        self.T_arr = self.T0 * np.ones(self.Nx)
        self.phase_arr = np.zeros(self.Nx)  # 0: solid; 1: s-l mixture; 2: liquid; 3: gas.
        self.dL = 0
        self.fp_arr = np.full(self.Nx, np.nan)
        self.f_mat = np.full((self.Ns, self.Nx), np.nan)
        self.record_state()

        # state variable initialization
        self.t = 0.0

    def record_state(self):
        self.T_mat.append(self.T_arr.copy())
        self.phase_mat.append(self.phase_arr.copy())
        self.dL_arr.append(self.dL)
        self.fp_mat.append(self.fp_arr.copy())
        self.f_ten.append(self.f_mat.copy())

    def get_P_sat(self, T):
        Ps = []
        for sp_name in self.sp_name_list:
            df = self.df_dict[sp_name]
            dB = df["dB"]
            Tb = df["bp"]
            rhs = (4.1012 + dB) * (T / Tb - 1) / (T / Tb - 1 / 8)
            Ps.append(10 ** rhs * cst.atm)
        Ps = np.array(Ps)
        return Ps

    def get_H_vap(self, T):
        H_vap = []
        for sp_name in self.sp_name_list:
            df = self.df_dict[sp_name]
            dB = df["dB"]
            Tb = df["bp"]
            drhs_dt = 56 * (4.1012 + dB) * Tb / (8 * T - Tb) ** 2
            H_vap.append(np.log(10) * drhs_dt * cst.gas_constant * T ** 2)
        H_vap = np.array(H_vap)
        return H_vap

    def get_evaporation_rate(self, T, f_prod, f_polymer):
        total_f = np.sum(f_prod, axis=0) + f_polymer
        x = np.zeros((self.Ns, len(T)))
        for i in range(len(T)):
            if total_f[i] > 0:
                x[:, i] = f_prod[:, i] / total_f[i]
        self.P_sat = self.get_P_sat(T)
        return x * self.P_sat * self.eta * self.rho_l * np.sqrt(1 / (2 * np.pi * cst.gas_constant * T * self.MW_arr.reshape(-1, 1)))

    def main(self):
        self.initialize()
        self.save_case_dict()

        rho_dict = {0: self.rho_s, 1: 0.5 * (self.rho_s + self.rho_l), 2: self.rho_l, 3: np.nan}
        k_dict = {0: self.k_s, 1: 0.5 * (self.k_s + self.k_l), 2: self.k_l, 3: np.nan}
        store_arr = np.zeros(self.Nx)  # energy in dT stored at phase change from solid to liquid

        for ti in tqdm(range(1, self.t_num), miniters=(self.t_num - 1) // 100):
            T_new = np.full(self.Nx, np.nan)
            fp_new = np.full(self.Nx, np.nan)
            f_new = np.full((self.Ns, self.Nx), np.nan)
            fe_new = np.full((self.Ns + 1, self.Nx), np.nan)

            rho_arr = np.array([rho_dict[p] for p in self.phase_arr])
            k_arr = np.array([k_dict[p] for p in self.phase_arr])

            liquid_ind = np.where(self.phase_arr == 2)[0]
            not_gas_ind = np.where(self.phase_arr != 3)[0]
            if len(not_gas_ind) == 0:
                print('All gas!')
                break
            boundary_ind = not_gas_ind[0]

            self.Ei = np.zeros((self.Ns, len(liquid_ind[1:-1])))
            if len(liquid_ind) > 3:
                for ind in liquid_ind:
                    if np.isnan(self.fp_arr[ind]):
                        self.fp_arr[ind] = rho_arr[ind] / self.MW
                        self.f_mat[:, ind] = 0
                inner_ind = liquid_ind[1:-1]

                def fe_rate(t, fe_in):
                    fe_top = np.concatenate([[self.fp_arr[liquid_ind[0]]], self.f_mat[:, liquid_ind[0]]])
                    fe_bottom = np.concatenate([[self.fp_arr[liquid_ind[-1]]], self.f_mat[:, liquid_ind[-1]]])
                    used_fe = np.hstack([fe_top.reshape(-1, 1), fe_in, fe_bottom.reshape(-1, 1)])

                    kr = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.T_arr[inner_ind]))
                    rxn_rate = kr * fe_in[0, :] * 2 / self.N
                    alpha_i = np.array([self.x_reaction(temp) for temp in self.T_arr[inner_ind]]).T
                    beta_i = self.N * alpha_i / np.sum([(i + 1) * alpha_i[i, :] for i in range(alpha_i.shape[0])])
                    self.Ei = self.get_evaporation_rate(self.T_arr[inner_ind], fe_in[1:, :], fe_in[0, :])
                    source_i = beta_i * rxn_rate - self.Ei
                    source = np.vstack([-rxn_rate.reshape(1, -1), source_i])
                    tmp_arr = np.concatenate([[self.D], self.D_arr]).reshape(-1, 1) / self.dx ** 2 * (used_fe[:, 2:] - 2 * fe_in + used_fe[:, :-2]) + source
                    return tmp_arr

                fe_mat = np.vstack([self.fp_arr.reshape(1, -1), self.f_mat])
                fe_new[:, inner_ind] = advance_rk4(fe_rate, ti * self.dt, fe_mat[:, inner_ind], self.dt)
                fe_new[fe_new < 0] = 0
                fe_new[:, liquid_ind[0]] = fe_mat[:, liquid_ind[0]]
                fe_new[:, liquid_ind[-1]] = fe_mat[:, liquid_ind[-1]]
                fp_new = fe_new[0, :]
                f_new = fe_new[1:, :]

                self.Ei = self.get_evaporation_rate(self.T_arr[inner_ind], self.f_mat[:, inner_ind], self.fp_arr[inner_ind])
                self.dL += np.sum(self.Ei * self.dt * self.MW_arr.reshape(-1, 1) / self.rho_l * self.dx)

            inner_ind = not_gas_ind[1:-1]

            def T_rate(t, T_in):
                used_T = np.concatenate([[self.T_arr[boundary_ind]], T_in, [self.T_arr[-1]]])
                kr = np.zeros(len(inner_ind))
                kr[:self.Ei.shape[1]] = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * T_in[:self.Ei.shape[1]]))
                Q_rxn = self.dH * 2 * rho_arr[inner_ind] / self.MW * kr
                self.H_vap = self.get_H_vap(T_in[:self.Ei.shape[1]])
                evap_heat = np.zeros(len(inner_ind))
                evap_heat[:self.Ei.shape[1]] = np.sum(self.Ei * self.H_vap, axis=0)
                return (k_arr[inner_ind] / self.dx ** 2 * (used_T[2:] - 2 * T_in + used_T[:-2]) - Q_rxn - evap_heat) / (rho_arr[inner_ind] * self.cp)

            T_new[inner_ind] = advance_rk4(T_rate, ti * self.dt, self.T_arr[inner_ind], self.dt)
            T_new[-1] = T_new[-2]

            phase_change_arg = np.where(np.isin(self.phase_arr, [0, 1]) & (T_new > self.T_melt))[0]
            self.phase_arr[phase_change_arg] = 1
            extra_T_arr = T_new[phase_change_arg] - self.T_melt
            T_new[phase_change_arg] = self.T_melt
            store_arr[phase_change_arg] += extra_T_arr

            lh_T = self.lh / self.cp
            over_arg = np.intersect1d(np.where(store_arr >= lh_T)[0], phase_change_arg)
            self.phase_arr[over_arg] = 2
            T_new[over_arg] = self.T_melt + (store_arr[over_arg] - lh_T)

            T_new[boundary_ind] = T_new[boundary_ind + 1] + self.q0 / k_arr[boundary_ind] * self.dx
            if self.phase_arr[boundary_ind] in [0, 1] and T_new[boundary_ind] > self.T_melt:
                self.phase_arr[boundary_ind] = 1
                extra_T = T_new[boundary_ind] - self.T_melt
                T_new[boundary_ind] = self.T_melt
                store_arr[boundary_ind] += extra_T
                if store_arr[boundary_ind] >= lh_T:
                    self.phase_arr[boundary_ind] = 2
                    T_new[boundary_ind] = self.T_melt + (store_arr[boundary_ind] - lh_T)

            self.phase_arr[boundary_ind: int(np.round(self.dL / self.dx))] = 3
            self.T_arr = T_new.copy()
            self.fp_arr = fp_new.copy()
            self.f_mat = f_new.copy()
            if ti in self.t_arg_arr:
                self.record_state()

        self.save_result()

    def plot_box(self):
        fig, ax = plt.subplots()

        def animate(ti):
            plt.cla()
            x = self.x_arr * 100
            ax = plt.gca()

            plt.plot(1, 1, ">k", transform=ax.transAxes, clip_on=False)
            plt.plot(0, 0, "vk", transform=ax.transAxes, clip_on=False)
            plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.spines[["bottom", "right"]].set_visible(False)

            plt.xlabel("Temperature, $T$ (K)")
            plt.ylabel("Depth, $x$ (cm)")
            ax.xaxis.set_label_position("top")

            temp_diff = self.Ts - self.T0
            temp_s1 = self.T0 - 0.1 * temp_diff
            temp_s2 = self.Ts + 0.1 * temp_diff
            temp_l1 = self.T0 - 0.2 * temp_diff
            temp_l2 = self.Ts + 0.2 * temp_diff
            l_s1 = 0
            l_l1 = -1.1 * self.L
            l_l2 = 1.1 * self.L
            temp_lim = [temp_l1, temp_l2]
            depth_lim = [l_l1, l_l2]
            box_width_lim = [temp_s1, temp_s2]
            ax.spines["left"].set_bounds(l_s1, l_l2)
            ax.spines["top"].set_bounds(temp_s1, temp_l2)

            color_list = ["gray", "g", "dodgerblue", "paleturquoise"]
            lb_list = ["Solid", "S-L Mixture", "Liquid", "Gas"]
            for ind, (color, lb) in enumerate(zip(color_list, lb_list)):
                pos = np.where(self.phase_mat[ti, :] == ind)[0]
                if len(pos) != 0:
                    plt.fill_between(box_width_lim, x[np.max(pos)], x[np.min(pos)], color=color, label=lb)
                else:
                    plt.fill_between([], [], [], color=color, label=lb)
            plt.plot(self.T_mat[ti, :], x, color="darkred", label="Temperature")
            plt.xlim(temp_lim)
            plt.ylim(depth_lim)
            ax.invert_yaxis()
            plt.text(0.3, 0.01, "{:.1f} s".format(self.t_store_arr[ti]), transform=ax.transAxes, color="k")
            plt.legend(loc="lower right")

            plt.tight_layout()

        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=range(0, self.phase_mat.shape[0], 10))
        writer = matplotlib.animation.FFMpegWriter(fps=100)
        ani.save("{}/Box.mp4".format(self.folder), writer=writer)

    def cal_steady_state(self):
        c1 = 2 * self.dH * self.rho_l * self.A_beta * cst.gas_constant * self.gamma / (self.MW * self.k_l * self.Ea)
        c3 = cst.gas_constant * self.T_melt / self.Ea
        c4 = 1 + self.dH / (self.MW0 * self.lh)

        g = lambda u: scipy.special.expi(-1 / u) + u * np.exp(-1 / u)
        gc2 = (self.q0 * cst.gas_constant / (self.k_l * self.Ea)) ** 2 * (1 - 1 / c4 ** 2) / (2 * c1) + g(c3)
        c2 = 1.0
        for _ in range(100):
            c2 = c2 - (g(c2) - gc2) / np.exp(-1 / c2)
        self.Ts = c2 * self.Ea / cst.gas_constant

        b0 = (g(c2) - c4 ** 2 * g(c3)) / (c4 ** 2 - 1)
        h = lambda y: 1 / np.sqrt(g(y) + b0)
        self.Lm = 1 / np.sqrt(2 * c1) * scipy.integrate.quad(h, c3, c2)[0]

        self.rb = self.k_l * self.Ea / (self.lh * self.rho_l * cst.gas_constant) * np.sqrt(2 * c1 * (g(c2) - g(c3)) / (c4 ** 2 - 1))

    def plot_properties(self):
        bi_arr = []
        for ti in tqdm(range(self.phase_mat.shape[0])):
            for xi in range(self.x_num):
                if self.phase_mat[ti, xi] != 3:
                    bi_arr.append(xi)
                    break
        bi_arr = np.array(bi_arr)
        Ts_arr = []
        for ti, bi in enumerate(tqdm(bi_arr)):
            Ts_arr.append(self.T_mat[ti, bi])
        Ts_arr = np.array(Ts_arr)
        plt.figure()
        plt.plot(self.t_store_arr, Ts_arr, color="k", label="Numerical transient")
        plt.xlabel("Time (s)")
        plt.ylabel("Surface temperature (K)")
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, self.Ts * np.ones(2), color="red", ls="--", label="Theoretical steady-state")
        plt.xlim(xlim)
        plt.legend(loc="lower right")
        plt.title("(a) Surface temperature")
        plt.savefig("{}/Surface_Temperature.png".format(self.folder), bbox_inches="tight")

        thick_arr = []
        for ti in tqdm(range(self.phase_mat.shape[0])):
            thick_arr.append(len(np.where(self.phase_mat[ti, :] == 2)[0]) * self.dx)
        thick_arr = np.array(thick_arr)
        plt.figure()
        plt.plot(self.t_store_arr, thick_arr * 100, color="k", label="Numerical transient")
        plt.xlabel("Time (s)")
        plt.ylabel("Molten layer thickness (cm)")
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, self.Lm * 100 * np.ones(2), color="red", ls="--", label="Theoretical steady-state")
        plt.xlim(xlim)
        plt.legend(loc="lower right")
        plt.title("(b) Molten layer thickness")
        plt.savefig("{}/Thickness.png".format(self.folder), bbox_inches="tight")

        plt.figure()
        y = self.x_arr[bi_arr] * 100
        plt.plot(self.t_store_arr, y, color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("Surface position (cm)")
        arg = np.where((self.t_store_arr > 0.8 * self.t_end) & (self.t_store_arr < self.t_end))[0]
        m, b = np.polyfit(self.t_store_arr[arg], y[arg], deg=1)
        plt.plot(self.t_arr, m * self.t_arr + b, color="r", lw=1)
        plt.text(0.4, 0.3, "$y$ = {:.4f} $x$ + {:.4f}".format(m, b), transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, r"Numerical slope = {:.1f} {}m/s".format(m * 1e4, mu_sb), transform=plt.gca().transAxes)
        plt.text(0.05, 0.78, r"Theoretial $r_b$ = {:.1f} {}m/s".format(self.rb * 1e6, mu_sb), transform=plt.gca().transAxes)
        plt.title("(c) Surface regression")
        plt.savefig("{}/Regression_Rate.png".format(self.folder), bbox_inches="tight")


def anchor_point():
    pass


if __name__ == '__main__':
    tt = Polymer()
    tt.folder = "{}/output/integrated/Case6".format(work_dir)
    tt.db_path = '{}/data/polymer_evaporation.xlsx'.format(work_dir)
    tt.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer"]
    tt.x_reaction = lambda T: np.array([33288 * T ** -2.174, 714581 * T ** -2.743, 1 - 33288 * T ** -2.174 - 714581 * T ** -2.743])
    tt.x_reaction_str = 'lambda T: np.array([33288 * T ** -2.174, 714581 * T ** -2.743, 1 - 33288 * T ** -2.174 - 714581 * T ** -2.743])'
    # tt.x_reaction = lambda T: np.array([487185 * T ** -2.413, 6e9 * T ** -4.192, 1 - 487185 * T ** -2.413 - 6e9 * T ** -4.192])
    # tt.x_reaction_str = 'lambda T: np.array([487185 * T**-2.413, 6e9 * T**-4.192, 1 - 487185 * T**-2.413 - 6e9 * T**-4.192])'
    tt.T0 = 300  # K, initial temperature
    tt.T_melt = 165 + 273  # K, melting temperature
    tt.m_polymer_init = 10e-3  # kg, initial mass of polymer
    tt.eta = 7.4e-6  # m2/kg, effective surface area coefficient
    tt.lumped_A = 2e11  # 1/s, lumped pre-exponential factor for polymer decomposition
    tt.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    tt.q0 = 1e5  # W/m2, heat flux from gas phase
    tt.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_s = 1.42e3  # kg/m3, solid phase density
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.lh = 150e3  # J/kg, latent heat of fuel melting
    tt.N = 3000  # number of polymer degree
    tt.D = 2e-15  # m2/s, polymer diffusion coefficient
    tt.L = 2e-2  # m
    tt.t_end = 800  # s
    tt.Nx = 500 + 1
    tt.t_num = 1000000 + 1
    tt.t_store = 100
    tt.main()

    pass
