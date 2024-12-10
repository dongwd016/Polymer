import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd
import scipy
from scipy import constants as cst
from tqdm.auto import tqdm
from tqdm.contrib import tzip

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


class Evaporation:
    def __init__(self):
        self.save_list = ['folder', 'db_path', 'sp_name_list', 'sp_num', 'x_reaction', 'm_polymer_init', 'S', 'P', 'lumped_A', 'lumped_Ea', 't_end', 't_num', 'temp_control',
                          'n_threshold']
        self.result_list = ['n_mat', 'm_polymer_arr', 'D_mat', 'C_mat', 't_arr', 'temp_arr']

        self.folder = None
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
            np.save("{}/{}.npy".format(self.folder, name), getattr(self, name))

    def initialize(self):
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        if self.db is None:
            self.db = pd.read_excel(self.db_path)
        self.df_dict = {df["Name"]: df for _, df in self.db.iterrows()}
        self.MW_arr = np.array([self.df_dict[sp]['MW'] for sp in self.sp_name_list])
        self.sp_num = len(self.sp_name_list)

        self.n_mat = []
        self.m_polymer_arr = []
        self.D_mat = []
        self.C_mat = []
        self.t_arr = []
        self.temp_arr = []

        self.n_arr = np.zeros(self.sp_num)
        self.m_polymer = self.m_polymer_init
        self.D_arr = np.zeros(self.sp_num)
        self.C_arr = np.zeros(self.sp_num)
        self.t = 0.0
        self.temp = self.temp_control(self.t)
        self.record_state()

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

    def get_evaporation_rate(self, T, n_prod):
        total_n = np.sum(n_prod) + self.m_polymer / np.sum(self.MW_arr * self.x_reaction(self.temp))
        if total_n > 0:
            x = n_prod / total_n
        else:
            return np.zeros(self.sp_num)
        self.P_sat = self.get_P_sat(T)
        D = x * self.P_sat * self.S * np.sqrt(1 / (2 * np.pi * cst.gas_constant * T * self.MW_arr))
        return D

        # y = x * self.P_sat / self.P
        # D1 = self.P * self.S * np.sqrt(1 / (2 * np.pi * cst.gas_constant * T * self.MW_arr))
        #
        # b = self.get_P_sat(T - 20) / self.P
        # left, right = 0.0, 1.0
        # f = lambda q: np.sum(y / (q + b * (1 - q))) - 1.0
        # val_left, val_right = f(left), f(right)
        # if val_left * val_right > 0:
        #     raise Exception('Bisection same sign!')
        # while right - left > 1e-6:
        #     mid = (left + right) / 2
        #     val_mid = f(mid)
        #     if val_mid * val_left < 0:
        #         right = mid
        #     else:
        #         left = mid
        # RD = mid
        # RD = 0.0
        #
        # u = y / (RD + b * (1 - RD))
        # R2 = D1 * RD
        # Df = y * D1 - u * R2
        # return Df

    def record_state(self):
        self.n_mat.append(self.n_arr.copy())
        self.m_polymer_arr.append(self.m_polymer)
        self.D_mat.append(self.D_arr.copy())
        self.C_mat.append(self.C_arr.copy())
        self.t_arr.append(self.t)
        self.temp_arr.append(self.temp)

    def main(self):
        self.initialize()
        self.save_case_dict()

        # def n_rate(t, n):
        #     m_polymer = n[0]
        #     n_prod = n[1:]
        #     dn_dt = np.zeros(self.sp_num)
        #     temp = self.T_control(t)
        #     k_reaction = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * temp))
        #     dn_dt[0] = -k_reaction * m_polymer
        #     sto = self.x_reaction(temp)
        #     dn_dt[1:] = sto * k_reaction * m_polymer / np.sum(self.MW_arr * sto) - self.get_evaporation_rate(temp, n_prod)
        #     return dn_dt

        outer_t_arr = np.linspace(0, self.t_end, self.t_num)
        dt_arr = np.diff(outer_t_arr)
        for ind, (ti, dt) in enumerate(tzip(outer_t_arr, dt_arr)):
            if ind == 215:
                a = 1
            if self.m_polymer == 0 and np.sum(self.n_arr) == 0:
                self.t = ti
                self.temp = self.temp_control(self.t)
                self.D_arr = np.zeros(self.sp_num)
                self.record_state()
                continue

            cur_dt = dt
            k_reaction = self.lumped_A * np.exp(-self.lumped_Ea / (cst.gas_constant * self.temp))
            if dt >= 1 / k_reaction and self.m_polymer > 0:
                cur_dt = 1 / k_reaction
            self.m_polymer -= k_reaction * self.m_polymer * cur_dt
            sto = self.x_reaction(self.temp)
            self.n_arr += sto * k_reaction * self.m_polymer / np.sum(self.MW_arr * sto) * cur_dt

            while True:
                self.D_arr = self.get_evaporation_rate(self.temp, self.n_arr)
                tmp_n = self.n_arr - self.D_arr * cur_dt
                if np.min(tmp_n) < -self.n_threshold:
                    arg = np.argmin(tmp_n)
                    cur_dt = self.n_arr[arg] / self.D_arr[arg]
                self.n_arr -= self.D_arr * cur_dt
                self.n_arr[self.n_arr < self.n_threshold] = 0
                self.C_arr += self.D_arr * cur_dt
                self.t = self.t_arr[-1] + cur_dt
                self.temp = self.temp_control(self.t)
                self.record_state()

                if self.t == ti:
                    break
                else:
                    cur_dt = ti - self.t
        self.save_result()


def anchor_point():
    pass


if __name__ == '__main__':
    e = Evaporation()
    e.folder = '{}/output/evaporation/Case19'.format(work_dir)
    e.db_path = '{}/data/polymer_evaporation.xlsx'.format(work_dir)
    e.sp_name_list = ["Styrene", "Styrene dimer", "Styrene trimer", "Styrene 4-mer", "Styrene 5-mer"]
    e.x_reaction = lambda temp: np.array([8, 4, 2, 1, 1], dtype=float)

    e.m_polymer_init = 10e-3  # kg, initial mass of polymer
    # e.S = e.m_polymer_init * 7.4e-6  # m2, effective surface area
    e.S = 3e-4  # m2, surface area
    # e.m_polymer_init = 0.5  # kg, initial mass of polymer
    # e.S = 1e-6  # m2, surface area

    e.P = cst.atm  # Pa, ambient pressure
    e.lumped_A = 2e11  # 1/s, lumped pre-exponential factor for polymer decomposition
    e.lumped_Ea = 43e3 * cst.calorie  # J/mol, lumped activation energy for polymer decomposition
    T_min, T_max = 323, 873
    beta = 100 / 60  # K/s, heating rate
    e.t_end = (T_max - T_min) / beta  # s, simulation time
    e.t_num = 1000  # number of sample time
    e.temp_control = lambda t: T_min + beta * t  # K, function of time, controlled temperature profile
    # e.n_threshold = np.finfo(float).eps
    e.n_threshold = 1e-30
    e.main()
    pass
