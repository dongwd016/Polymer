import os
import json

import numpy as np
import scipy
from scipy import constants as cst
from tqdm.auto import tqdm
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
                          't_end', 'x_num', 't_num', 't_store', 'dt', 'dx', 'cfl', 'db_path', 'sp_name_list', 'sp_num', 'x_reaction', 'm_polymer_init', 'S', 'P', 'lumped_A',
                          'lumped_Ea', 't_end', 't_num', 'temp_control', 'n_threshold']
        self.result_list = ['x_arr', 't_arr', 't_arg_arr', 't_store_arr', 'T_mat', 'phase_mat', 'dL_arr', 'n_mat', 'm_polymer_arr', 'D_mat', 'C_mat', 't_arr', 'temp_arr']

        self.folder = None

        # Transient related (non-uniform temperature)
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

        self.L = None  # m
        self.t_end = None  # s
        self.x_num = None
        self.t_num = None
        self.t_store = None
        self.x_arr = None
        self.t_arr = None
        self.t_arg_arr = None
        self.t_store_arr = None
        self.dt = None
        self.dx = None
        self.cfl = None  # pseudo CFL number, alpha * dt / dx ** 2

        self.T_mat = None
        self.phase_mat = None
        self.dL_arr = None

        # Evaporation related
        self.db_path = None
        self.sp_name_list = None  # [K,], decomposition product name list
        self.sp_num = None  # number of decomposition products
        self.x_reaction = None  # [K,], function of temperature, polymer decomposition product mole fraction array directly from reaction
        self.m_polymer_init = None  # kg, initial mass of polymer
        self.S = None  # m2, surface area
        self.P = None  # Pa, ambient pressure
        self.lumped_A = None  # 1/s, lumped pre-exponential factor for polymer decomposition
        self.lumped_Ea = None  # J/mol, lumped activation energy for polymer decomposition
        self.t_end = None  # s, simulation time
        self.t_num = None  # number of sample time
        self.temp_control = None  # K, function of time, controlled temperature profile
        self.n_threshold = None  # n smaller than this will be considered as 0

        self.n_mat = None  # mol [N,K], time history of number of mole array of liquid phase decomposition products
        self.n_arr = None  # mol [K,], current number of mole array of liquid phase decomposition products
        self.m_polymer_arr = None  # kg [K,], polymer mass time history
        self.m_polymer = None  # kg, remaining polymer mass
        self.D_mat = None  # mol/s [N,K], time history of evaporation rate of each product
        self.D_arr = None  # mol/s [K,], evaporation rate of each product
        self.C_mat = None  # mol [N,K], time history of C_arr
        self.C_arr = None  # mol [K,], number of mole of each product being collected in the gas phase cumulatively
        self.t_arr = None  # s [K,], result time array
        self.t = None  # s, current time
        self.temp_arr = None  # K [K,], temperature time history
        self.temp = None  # K, current temperature

        self.db = None  # database recording species properties
        self.df_dict = None  # df_dict generated from db
        self.MW_arr = None  # kg/mol [K,] molecular weight array of each product
        self.P_sat = None  # Pa, [K,] vapor pressure array of each product

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

        self.x_arr = np.linspace(0, self.L, self.x_num)
        self.t_arr = np.linspace(0, self.t_end, self.t_num)
        self.t_arg_arr = np.arange(0, self.t_num, self.t_store, dtype=int)
        self.t_store_arr = self.t_arr[self.t_arg_arr]
        self.dt = self.t_arr[1] - self.t_arr[0]
        self.dx = self.x_arr[1] - self.x_arr[0]
        self.cfl = self.k_l / (self.rho_l * self.cp) * self.dt / self.dx ** 2
        print('alpha * dt / dx^2 = {}'.format(self.cfl))

        self.T_mat = np.empty(((self.t_num - 1) // self.t_store + 1, self.x_num))
        self.phase_mat = np.empty(((self.t_num - 1) // self.t_store + 1, self.x_num))
        self.dL_arr = np.empty((self.t_num - 1) // self.t_store + 1)

    def main(self):
        self.initialize()
        self.save_case_dict()

        T_old = self.T0 * np.ones(self.x_num)
        T_old[0] = T_old[1] + self.q0 / self.k_s * self.dx
        self.T_mat[0, :] = T_old

        phase_arr = np.zeros(self.x_num)  # 0: solid; 1: s-l mixture; 2: liquid; 3: gas.
        self.phase_mat[0, :] = phase_arr
        bi = 0
        rho_dict = {0: self.rho_s, 1: 0.5 * (self.rho_s + self.rho_l), 2: self.rho_l, 3: np.nan}
        k_dict = {0: self.k_s, 1: 0.5 * (self.k_s + self.k_l), 2: self.k_l, 3: np.nan}
        store_arr = np.zeros(self.x_num)  # energy in dT stored at phase change from solid to liquid

        dL = 0
        t_ind = 0
        for ti in tqdm(range(1, self.t_num), miniters=(self.t_num - 1) // 100):
            for pi in range(bi, self.x_num):
                if phase_arr[pi] != 3:
                    bi = pi
                    break

            rho_arr = np.array([rho_dict[p] for p in phase_arr])
            k_arr = np.array([k_dict[p] for p in phase_arr])

            liq_arg = np.where(phase_arr[bi + 1: -1] == 2)[0]

            def k_rxn(T):
                tmp_arr = np.zeros_like(T)
                tmp_arr[liq_arg] = self.A_beta * np.exp(-self.Ea / (cst.gas_constant * T[liq_arg]))
                return tmp_arr

            Q_rxn = lambda T: self.dH * 2 * rho_arr[bi + 1: -1] / self.MW * k_rxn(T) * self.gamma
            dL += np.sum(2 * k_rxn(T_old[bi + 1: -1]) * self.gamma * self.MW0 / self.MW * self.dt * self.dx)

            def T_rate(t, T_in):
                used_T = np.concatenate([[T_old[bi]], T_in, [self.T0]])
                tmp_arr = (k_arr[bi + 1: -1] / self.dx ** 2 * (used_T[2:] - 2 * T_in + used_T[:-2]) - Q_rxn(T_in)) / (rho_arr[bi + 1: -1] * self.cp)
                return tmp_arr

            T_new = np.empty(self.x_num)
            T_new.fill(np.nan)
            T_new[-1] = self.T0
            T_new[bi + 1: -1] = advance_rk4(T_rate, ti * self.dt, T_old[bi + 1: -1], self.dt)

            phase_change_arg = np.where(np.isin(phase_arr, [0, 1]) & (T_new > self.T_melt))[0]
            phase_arr[phase_change_arg] = 1
            extra_T_arr = T_new[phase_change_arg] - self.T_melt
            T_new[phase_change_arg] = self.T_melt
            store_arr[phase_change_arg] += extra_T_arr

            lh_T = self.lh / self.cp
            over_arg = np.intersect1d(np.where(store_arr >= lh_T)[0], phase_change_arg)
            phase_arr[over_arg] = 2
            T_new[over_arg] = self.T_melt + (store_arr[over_arg] - lh_T)

            T_new[bi] = T_new[bi + 1] + self.q0 / k_arr[bi] * self.dx
            if phase_arr[bi] in [0, 1] and T_new[bi] > self.T_melt:
                phase_arr[bi] = 1
                extra_T = T_new[bi] - self.T_melt
                T_new[bi] = self.T_melt
                store_arr[bi] += extra_T
                if store_arr[bi] >= lh_T:
                    phase_arr[bi] = 2
                    T_new[bi] = self.T_melt + (store_arr[bi] - lh_T)

            phase_arr[bi: int(np.round(dL / self.dx))] = 3

            if ti in self.t_arg_arr:
                t_ind += 1
                self.phase_mat[t_ind, :] = phase_arr
                self.T_mat[t_ind, :] = T_new
                self.dL_arr[t_ind] = dL
            T_old = T_new

        self.save_result()

    def main_TGA(self):
        self.initialize()
        self.save_case_dict()

        T_old = self.T0 * np.ones(self.x_num)
        T_old[0] = T_old[1] + self.q0 / self.k_l * self.dx
        self.T_mat[0, :] = T_old

        bi = 0
        dL = 0
        t_ind = 0
        for ti in tqdm(range(1, self.t_num)):
            rho_arr = self.rho_l * np.ones(self.x_num)
            k_arr = self.k_l * np.ones(self.x_num)

            def k_rxn(T):
                return self.A_beta * np.exp(-self.Ea / (cst.gas_constant * T))

            Q_rxn = lambda T: self.dH * 2 * rho_arr[bi + 1: -1] / self.MW * k_rxn(T) * self.gamma
            dL += np.sum(2 * k_rxn(T_old[bi:]) * self.gamma * self.MW0 / self.MW * self.dt * self.dx)

            def T_rate(t, T_in):
                used_T = np.concatenate([[T_old[bi]], T_in, [T_old[-1]]])
                tmp_arr = (k_arr[bi + 1: -1] / self.dx ** 2 * (used_T[2:] - 2 * T_in + used_T[:-2]) - Q_rxn(T_in)) / (rho_arr[bi + 1: -1] * self.cp)
                return tmp_arr

            T_new = np.empty(self.x_num)
            T_new.fill(np.nan)
            T_new[-1] = self.T0 + self.slope_Tb * ti * self.dt
            T_new[bi + 1: -1] = advance_rk4(T_rate, ti * self.dt, T_old[bi + 1: -1], self.dt)
            T_new[bi] = T_new[bi + 1] + self.q0 / k_arr[bi] * self.dx

            if ti in self.t_arg_arr:
                t_ind += 1
                self.T_mat[t_ind, :] = T_new
                self.dL_arr[t_ind] = dL
            T_old = T_new

            bi = int(np.round(dL / self.dx))
            if bi >= self.x_num - 1:
                self.T_mat = self.T_mat[: t_ind + 1, :]
                self.dL_arr = self.dL_arr[: t_ind + 1]
                break

        self.save_result()

    def main_TGA_top_heat(self):
        self.initialize()
        self.save_case_dict()

        T_old = self.T0 * np.ones(self.x_num)
        T_old[0] = T_old[1] + self.q0 / self.k_l * self.dx
        self.T_mat[0, :] = T_old

        bi = 0
        dL = 0
        t_ind = 0
        for ti in tqdm(range(1, self.t_num)):
            rho_arr = self.rho_l * np.ones(self.x_num)
            k_arr = self.k_l * np.ones(self.x_num)

            def k_rxn(T):
                return self.A_beta * np.exp(-self.Ea / (cst.gas_constant * T))

            Q_rxn = lambda T: self.dH * 2 * rho_arr[bi + 1: -1] / self.MW * k_rxn(T) * self.gamma
            dL += np.sum(2 * k_rxn(T_old[bi:]) * self.gamma * self.MW0 / self.MW * self.dt * self.dx)

            def T_rate(t, T_in):
                used_T = np.concatenate([[T_old[bi]], T_in, [T_old[-1]]])
                tmp_arr = (k_arr[bi + 1: -1] / self.dx ** 2 * (used_T[2:] - 2 * T_in + used_T[:-2]) - Q_rxn(T_in)) / (rho_arr[bi + 1: -1] * self.cp)
                return tmp_arr

            T_new = np.empty(self.x_num)
            T_new.fill(np.nan)
            T_new[bi] = T_old[bi + 1] + self.q0 / k_arr[bi] * self.dx
            T_new[bi + 1: -1] = advance_rk4(T_rate, ti * self.dt, T_old[bi + 1: -1], self.dt)
            T_new[-1] = T_new[-2]

            if ti in self.t_arg_arr:
                t_ind += 1
                self.T_mat[t_ind, :] = T_new
                self.dL_arr[t_ind] = dL
            T_old = T_new

            bi = int(np.round(dL / self.dx))
            if bi >= self.x_num - 1:
                self.T_mat = self.T_mat[: t_ind + 1, :]
                self.dL_arr = self.dL_arr[: t_ind + 1]
                break

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
    tt.folder = "{}/output/integrated/Case1".format(work_dir)
    tt.T0 = 438  # K
    tt.slope_Tb = 30 / 60  # K/s
    tt.q0 = 3e3  # W/m2
    tt.qb = 0  # W/m2
    tt.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    tt.rho_l = 1.2e3  # kg/m3, liquid phase density
    tt.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    tt.cv = 35 / tt.MW0  # J/kg·K
    tt.cp = tt.cv
    tt.A_beta = 1.8e13  # 1/s
    tt.Ea = 31.8 * cst.calorie * 1e3  # J/mol
    tt.dH = 56e3  # J/mol, heat absorbed by beta scission
    tt.MW = 1e2  # kg/mol, molecular weight of POM
    tt.gamma = 1
    tt.L = 1e-3  # m
    tt.t_end = 800  # s
    tt.x_num = 50 + 1
    tt.t_num = 1000000 + 1
    tt.t_store = 100
    tt.main_TGA_top_heat()

    pass
