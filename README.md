# Polymer: transient model of solid polymer fuel regression

Python code for a one-dimensional, transient model of a solid polymer fuel (e.g., polyoxymethylene, POM) heated at its surface, as in a hybrid rocket combustion chamber. The model couples heat conduction, melting, depolymerization kinetics, evaporation of the decomposition products, and a moving (regressing) surface. It extends the steady-state analytical model in [SS_Model](https://github.com/dongwd016/SS_Model) and the counterflow regression-rate method in [Counterflow_Polymer](https://github.com/dongwd016/Counterflow_Polymer).

Developed by Wendi Dong in Prof. Hai Wang's group, Mechanical Engineering, Stanford University. This is research code; the cases in the scripts are the ones run during development.

## Contents

| File | Description |
| --- | --- |
| `src/transient.py` | `Transient`: conduction through the solid and melt layers with a beta-scission heat sink in the melt and a regressing surface. Central differences in space, classical RK4 in time (method of lines). `cal_steady_state()` evaluates the steady-state analytical solution (surface temperature, melt-layer thickness, regression rate) for comparison; `main_TGA()` and `main_TGA_top_heat()` simulate thermogravimetric analysis (TGA) heating. |
| `src/evaporation.py` | `Evaporation`: evaporation of multicomponent decomposition products (e.g., styrene oligomers) from a heated sample, with vapor pressures from the Nannoolal correlation and a Hertz-Knudsen evaporation flux. |
| `src/integrated.py` | `Polymer`: the integrated model. Conduction, melting, lumped depolymerization kinetics, multicomponent products with evaporation, and a time-dependent surface heat flux; checkpoints allow long runs to restart. `case42()` ... `case60()` are the development cases. |
| `src/counterflow_SS.py` | Couples the condensed phase to a Cantera counterflow diffusion flame through a surface energy balance (conduction, multicomponent Stefan-Maxwell diffusion, Soret effect, radiation) and solves for the surface temperature by bisection, giving the regression rate as a function of oxidizer velocity. |

The solver was verified against the analytical solutions and compared with TGA and counterflow-flame measurements (the comparison data are not included here).

## Data (`src/data/`)

- `POM.json`: POM properties used by `case58()` ... `case60()`.
- `polymer_evaporation.xlsx`: boiling points (several estimation methods), Nannoolal parameters, molecular weights, and diffusion coefficients of decomposition products.
- `FFCM2_CH2O.yaml`: gas-phase kinetics, the formaldehyde sub-model of FFCM-2 (13 species, 47 reactions), used by `counterflow_SS.py`.
- `TdependentStoicCoeff.xlsx`: reference data, temperature-dependent product distribution of polystyrene pyrolysis (literature measurements and the CRECK model); not read by the scripts.

## Usage

```bash
pip install -r requirements.txt
python src/integrated.py      # runs case60(); about 2 minutes on a laptop
```

To run another case, change the call in the `if __name__ == "__main__":` block at the end of the script. `transient.py`, `evaporation.py`, and `counterflow_SS.py` are run the same way.

Inputs are read from `src/data/`; results are written to `src/output/<model>/<case>/` (git-ignored): `case_dict.json` with all inputs, NumPy arrays of the time history (e.g., `T_mat.npy` temperature field, `phase_mat.npy` phase field, `dL_arr.npy` regressed thickness), and `log.txt`. Saving animations (`Transient.plot_box()`) requires ffmpeg on the `PATH`.

Tested with Python 3.10, NumPy 1.24, SciPy 1.10, pandas 2.1, Matplotlib 3.7, and Cantera 2.6.
