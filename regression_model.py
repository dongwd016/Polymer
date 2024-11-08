import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy
from scipy import constants as cst
from tqdm.auto import tqdm

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


class Regression:
    def __init__(self):
        self.save_list = ['T0', 'q0', 'k_s', 'k_l', 'rho_s', 'rho_l', 'MW0', 'cv', 'cp', 'A_beta', 'Ea', 'dH', 'MW', 'gamma', 'lh', 'T_melt', 'L', 't_end', 'x_num', 't_num',
                          't_store', 'dt', 'dx', 'cfl', 'Ts', 'Lm', 'rb']
        self.result_list = ['x_arr', 't_arr', 't_arg_arr', 't_store_arr', 'T_mat', 'phase_mat']

        self.folder = None

        self.T0 = None  # K, room temperature
        self.q0 = None  # W/m2, heat flux from gas phase
        self.k_s = None  # W/m·K, solid phase thermal conductivity
        self.k_l = None  # W/m·K, liquid phase thermal conductivity
        self.rho_s = None  # kg/m3, solid phase density
        self.rho_l = None  # kg/m3, liquid phase density
        self.MW0 = None  # kg/mol, molecular weight of monomer
        self.cv = None  # J/kg·K, constant volume specific heat
        self.cp = None  # J/kg·K, constant pressure specific heat
        self.A_beta = None  # 1/s, pre-exponential factor of polymer decomposition
        self.Ea = None  # J/mol, activation energy of polymer decomposition
        self.dH = None  # J/mol, heat absorbed by beta scission
        self.MW = None  # kg/mol, molecular weight of polymer
        self.gamma = None  # pre-exponential factor correction factor
        self.lh = None  # J/kg, latent heat of melting
        self.T_melt = None  # K, melting point

        self.L = None  # m, domain size
        self.t_end = None  # simulation time
        self.x_num = None  # number of sample points on length axis
        self.t_num = None  # number of sample points on time axis
        self.t_store = None  # save every this number of time steps
        self.x_arr = None  # depth coordinate
        self.t_arr = None  # time array
        self.t_arg_arr = None  # saved time index
        self.t_store_arr = None  # saved time
        self.dt = None  # time step
        self.dx = None  # grid size
        self.cfl = None  # alpha * dt / dx^2, needs to be < 1.0

        self.T_mat = None  # temperature profile
        self.phase_mat = None  # 0: solid; 1: s-l mixture; 2: liquid; 3: gas.
        self.T_old = None  # previous temperature profile
        self.phase_arr = None  # previous phase profile
        self.bi = None  # boundary index
        self.rho_dict = None  # density for each phase
        self.k_dict = None  # thermal conductivity for each phase
        self.store_arr = None  # K, energy in the unit of dT stored at phase change from solid to liquid
        self.dL = None  # m, regressed length
        self.t_ind = None  # saved snapshot index

        self.Ts = None  # K, steady-state surface temperature
        self.Lm = None  # m, steady-state molten layer thickness
        self.rb = None  # m/s, steady-state regression rate

    def save_case_dict(self):
        case_dict = {}
        for name in self.save_list:
            case_dict[name] = getattr(self, name)
        json.dump(case_dict, open("{}/case_dict.json".format(self.folder), "w"), indent=4, default=json_convert)

    def save_result(self):
        for name in self.result_list:
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

        self.T_mat = np.empty(((self.t_num - 1) // self.t_store + 1, self.x_num))
        self.phase_mat = np.empty(((self.t_num - 1) // self.t_store + 1, self.x_num))

        self.T_old = self.T0 * np.ones(self.x_num)
        self.T_old[0] = self.T_old[1] + self.q0 / self.k_s * self.dx
        self.T_mat[0, :] = self.T_old

        self.phase_arr = np.zeros(self.x_num)
        self.phase_mat[0, :] = self.phase_arr
        self.bi = 0
        self.rho_dict = {0: self.rho_s, 1: 0.5 * (self.rho_s + self.rho_l), 2: self.rho_l, 3: np.nan}
        self.k_dict = {0: self.k_s, 1: 0.5 * (self.k_s + self.k_l), 2: self.k_l, 3: np.nan}
        self.store_arr = np.zeros(self.x_num)

        self.dL = 0
        self.t_ind = 0

    def main(self):
        self.initialize()
        self.cal_steady_state()
        self.save_case_dict()

        for ti in tqdm(range(1, self.t_num)):
            for pi in range(self.bi, self.x_num):
                if self.phase_arr[pi] != 3:
                    self.bi = pi
                    break

            rho_arr = np.array([self.rho_dict[p] for p in self.phase_arr])
            k_arr = np.array([self.k_dict[p] for p in self.phase_arr])

            liq_arg = np.where(self.phase_arr[self.bi + 1: -1] == 2)[0]

            def k_rxn(T):
                tmp_arr = np.zeros_like(T)
                tmp_arr[liq_arg] = self.A_beta * np.exp(-self.Ea / (cst.gas_constant * T[liq_arg]))
                return tmp_arr

            Q_rxn = lambda T: self.dH * 2 * rho_arr[self.bi + 1: -1] / self.MW * k_rxn(T) * self.gamma
            self.dL += np.sum(2 * k_rxn(self.T_old[self.bi:]) * self.gamma * self.MW0 / self.MW * self.dt * self.dx)

            def T_rate(t, T_in):
                used_T = np.concatenate([[self.T_old[self.bi]], T_in, [self.T0]])
                tmp_arr = (k_arr[self.bi + 1: -1] / self.dx ** 2 * (used_T[2:] - 2 * T_in + used_T[:-2]) - Q_rxn(T_in)) / (rho_arr[self.bi + 1: -1] * self.cp)
                return tmp_arr

            T_new = np.empty(self.x_num)
            T_new.fill(np.nan)
            T_new[-1] = self.T0
            T_new[self.bi + 1: -1] = advance_rk4(T_rate, ti * self.dt, self.T_old[self.bi + 1: -1], self.dt)

            phase_change_arg = np.where(np.isin(self.phase_arr, [0, 1]) & (T_new > self.T_melt))[0]
            self.phase_arr[phase_change_arg] = 1
            extra_T_arr = T_new[phase_change_arg] - self.T_melt
            T_new[phase_change_arg] = self.T_melt
            self.store_arr[phase_change_arg] += extra_T_arr

            lh_T = self.lh / self.cp
            over_arg = np.intersect1d(np.where(self.store_arr >= lh_T)[0], phase_change_arg)
            self.phase_arr[over_arg] = 2
            T_new[over_arg] = self.T_melt + (self.store_arr[over_arg] - lh_T)

            T_new[self.bi] = T_new[self.bi + 1] + self.q0 / k_arr[self.bi] * self.dx
            if self.phase_arr[self.bi] in [0, 1] and T_new[self.bi] > self.T_melt:
                self.phase_arr[self.bi] = 1
                extra_T = T_new[self.bi] - self.T_melt
                T_new[self.bi] = self.T_melt
                self.store_arr[self.bi] += extra_T
                if self.store_arr[self.bi] >= lh_T:
                    self.phase_arr[self.bi] = 2
                    T_new[self.bi] = self.T_melt + (self.store_arr[self.bi] - lh_T)

            self.phase_arr[self.bi: int(np.round(self.dL / self.dx))] = 3

            if ti in self.t_arg_arr:
                self.t_ind += 1
                self.phase_mat[self.t_ind, :] = self.phase_arr
                self.T_mat[self.t_ind, :] = T_new
            self.T_old = T_new
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

            temp_lim = [250, 720]
            depth_lim = [-0.2, 2.2]
            box_width_lim = [280, 700]
            ax.spines["left"].set_bounds(0, 2.2)
            ax.spines["top"].set_bounds(280, 720)

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


if __name__ == '__main__':
    p = Regression()
    p.folder = '{}/output/regression/Case1'.format(work_dir)
    p.T0 = 300  # K
    p.q0 = 1e5  # W/m2, heat flux from gas phase
    p.k_s = 0.33  # W/m·K, solid phase thermal conductivity
    p.k_l = 0.14  # W/m·K, liquid phase thermal conductivity
    p.rho_s = 1.42e3  # kg/m3, solid phase density
    p.rho_l = 1.2e3  # kg/m3, liquid phase density
    p.MW0 = 30e-3  # kg/mol, CH2O molecular weight
    p.cv = 35 / p.MW0  # J/kg·K
    p.cp = p.cv
    p.A_beta = 1.8e13  # 1/s
    p.Ea = 30 * cst.calorie * 1e3  # J/mol
    p.dH = 56e3  # J/mol, heat absorbed by beta scission
    p.MW = 1e2  # kg/mol, molecular weight of POM
    p.gamma = 1
    p.lh = 150e3  # J/kg, latent heat of POM melting
    p.T_melt = 165 + 273  # K, POM melting point

    p.L = 0.02  # m
    p.t_end = 100  # s
    p.x_num = 2000 + 1
    p.t_num = 2000000 + 1
    p.t_store = 1000

    p.main()
    p.plot_box()
    p.plot_properties()
