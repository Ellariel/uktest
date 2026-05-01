import os
import copy
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from spreg import (
    ML_LagFE,
    ML_LagRE,
    ML_ErrorRE,
    ML_ErrorFE,
    PooledOLS,
    PanelFE,
    PanelRE,
)

from weights import compute_weights

warnings.simplefilter("ignore")

base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

print("data_dir:", data_dir)
print("results_dir:", results_dir)

df = pd.read_csv(os.path.join(data_dir, "df.csv"))
df = df.sort_values(by=["year", "name"])  # imortant for spreg!!!!
shape_data = (
    gpd.read_file(
        os.path.join(base_dir, "raw_data", "irradiance", "LAD_MAY_2024_UK_BFE.shp")
    )
    .set_crs("EPSG:27700")
    .to_crs("EPSG:4326")
)

models = {
    "ML_LagFE": ML_LagFE,
    "ML_LagRE": ML_LagRE,
    "ML_ErrorRE": ML_ErrorRE,
    "ML_ErrorFE": ML_ErrorFE,
    "PooledOLS": PooledOLS,
    "PanelFE": PanelFE,
    "PanelRE": PanelRE,
    "ML_LagFE(SLX)": ML_LagFE,
    "ML_LagRE(SLX)": ML_LagRE,
}
w_methods = ["inverse_distance", "k_nearest", "queen"]
y_vars = ["log_pv_cap", "log_pv_inst", "log_pv_cap_per_inst"]
x_vars = {
    "densities": ["pdensity", "hdensity", "fdensity", "adensity"],
    "capacities": ["log_gdp_cap", "log_fcapacity", "log_hcapacity", "log_acapacity"],
    "baseline": ["log_income", "log_hholds", "log_sunny_hours"],  # 'log_house_price',
}


def get_cached_weights(y, x, w_method, f):
    global shape_data, df
    if f in cache_weights:
        return cache_weights[f]
    else:
        w, X, Y = compute_weights(
            y, x, data=df, shape_data=shape_data, method=w_method, verbose=False
        )
        cache_weights[f] = (w, X, Y)
        return w, X, Y


def compute_model(m_method, w_method, y, x, f, fallback_x=None):
    w, X, Y = get_cached_weights(y, x, w_method, f)
    if w is not None:
        params = {
            "y": Y,
            "x": X,
            "w": w,
            "name_y": y,
            "name_x": x,
            "name_w": w_method,
            "slx_lags": 1 if "SLX" in m_method else 0,
        }
        try:
            return models[m_method.replace("(SLX)", "")](**params)
        except Exception as e:
            print(f"Error in {m_method}:{w_method}: {str(e)}")
            if fallback_x is not None:
                print(f"Falling back to {fallback_x} for {m_method}:{w_method}")
                return compute_model(
                    m_method,
                    w_method,
                    y,
                    fallback_x,
                    f + f" (fallback: {fallback_x})",
                    fallback_x=None,
                )
    return None


cache_weights = {}
weights_file = os.path.join(results_dir, "cache_weights.pickle")
if os.path.exists(weights_file):
    with open(weights_file, "rb") as f:
        cache_weights = pickle.load(f)


model_results = {}
results_file = os.path.join(results_dir, "model_results.pickle")
if os.path.exists(results_file):
    with open(results_file, "rb") as f:
        model_results = pickle.load(f)
else:
    for y in tqdm(y_vars):
        model_results.setdefault(y, {})
        for x_group, x_list in x_vars.items():
            model_results[y].setdefault(x_group, {})
            if x_group == "baseline":
                for m_name in models.keys():
                    model_results[y][x_group].setdefault(m_name, {})
                    for w_name in w_methods:
                        model_results[y][x_group][m_name].setdefault(w_name, {})
                        base_vars = copy.deepcopy(x_vars["baseline"])
                        fallback_vars = base_vars.copy()
                        fallback_vars.remove("log_sunny_hours")
                        f = f"Running {m_name}:{w_name} for {y} ~ ({x_group}): {base_vars}"
                        model_results[y][x_group][m_name][w_name][f] = compute_model(
                            m_name,
                            w_name,
                            y,
                            base_vars,  # if "FE" not in m_name else fallback_vars,
                            f,
                            fallback_x=fallback_vars,
                        )

            if x_group == "capacities":
                for m_name in models.keys():
                    model_results[y][x_group].setdefault(m_name, {})
                    for w_name in w_methods:
                        model_results[y][x_group][m_name].setdefault(w_name, {})
                        base_vars = copy.deepcopy(x_vars["baseline"])
                        fallback_vars = base_vars.copy()
                        fallback_vars.remove("log_sunny_hours")
                        f = f"Running {m_name}:{w_name} for {y} ~ ({x_group}): {x_vars['capacities']} + {base_vars}"
                        model_results[y][x_group][m_name][w_name][f] = compute_model(
                            m_name,
                            w_name,
                            y,
                            x_vars["capacities"] + base_vars,
                            # if "FE" not in m_name
                            # else x_vars["capacities"] + fallback_vars,
                            f,
                            fallback_x=x_vars["capacities"] + fallback_vars,
                        )

            if x_group == "densities":
                for m_name in models.keys():
                    model_results[y][x_group].setdefault(m_name, {})
                    for w_name in w_methods:
                        model_results[y][x_group][m_name].setdefault(w_name, {})
                        base_vars = copy.deepcopy(x_vars["baseline"])
                        fallback_vars = base_vars.copy()
                        fallback_vars.remove("log_sunny_hours")
                        for x in x_list:
                            f = f"Running {m_name}:{w_name} for {y} ~ ({x_group}): {x} + {base_vars}"
                            model_results[y][x_group][m_name][w_name][f] = (
                                compute_model(
                                    m_name,
                                    w_name,
                                    y,
                                    [x] + base_vars,
                                    # if "FE" not in m_name
                                    # else [x] + fallback_vars,
                                    f,
                                    fallback_x=[x] + fallback_vars,
                                )
                            )

    with open(results_file, "wb") as f:
        pickle.dump(model_results, f)

with open(weights_file, "wb") as f:
    pickle.dump(cache_weights, f)
