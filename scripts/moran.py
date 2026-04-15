import os
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from numbers import Number
from esda.moran import Moran
from scipy.stats import norm
import matplotlib.pyplot as plt

from weights import compute_weights

warnings.simplefilter("ignore")

base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

print("data_dir:", data_dir)
print("results_dir:", results_dir)

df = pd.read_csv(os.path.join(data_dir, "df.csv"))
df = df.sort_values(by=["year", "name"])
shape_data = (
    gpd.read_file(
        os.path.join(base_dir, "raw_data", "irradiance", "LAD_MAY_2024_UK_BFE.shp")
    )
    .set_crs("EPSG:27700")
    .to_crs("EPSG:4326")
)


def get_stars(p, p001="***", p01="**", p05="*", p10="+", p_=""):
    if not isinstance(p, Number):
        return p
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def average_morans_i(i, i_var, method="two-tailed"):
    """
    I: array of Moran's I values
    var_I: array of variances of Moran's I
    """
    i = np.array(i, dtype=float)
    var_i = np.array(i_var, dtype=float)
    w = 1 / var_i  # weights = inverse variance
    i_bar = np.sum(w * i) / np.sum(w)  # weighted mean
    se = np.sqrt(1 / np.sum(w))  # standard error
    z = i_bar / se  # test statistic
    p = 1 - norm.cdf(np.abs(z))  # p-value (two-tailed)
    p = p * 2 if method == "two-tailed" else p
    return {
        "i_mean": i_bar,
        "z": z,
        "se": se,
        "p-value": p,
    }


# Global spatial autocorrelation (Moran’s I)
methods = ["inverse_distance", "k_nearest", "queen"]
# variables = ["log_pv_cap", "log_pv_inst", "log_pv_cap_per_inst"]
variables = ["pv_cap", "pv_inst", "pv_cap_per_inst"]

moran_results_file = os.path.join(results_dir, "moran_results.pickle")
if not os.path.exists(moran_results_file):
    pv_cap_moran_results = []
    for y in df["year"].unique():
        df_y = df[df["year"] == y].copy()
        if len(df_y) > 0 and df_y[variables[0]].notna().sum() > 0:
            d = {"year": y}
            for m in methods:
                w, X, Y = compute_weights(
                    variables[0],
                    ["area"],
                    data=df_y,
                    shape_data=shape_data,
                    method=m,
                    verbose=False,
                )
                if w is not None:
                    mi = Moran(Y, w, transformation="R")
                    d[f"I_{m}"] = mi.I
                    d[f"p_{m}"] = mi.p_sim
                    d[f"VI_{m}"] = mi.VI_rand
            pv_cap_moran_results.append(d)
    pv_cap_moran_results = pd.DataFrame(pv_cap_moran_results)

    pv_inst_moran_results = []
    for y in df["year"].unique():
        df_y = df[df["year"] == y].copy()
        if len(df_y) > 0 and df_y[variables[1]].notna().sum() > 0:
            d = {"year": y}
            for m in methods:
                w, X, Y = compute_weights(
                    variables[1],
                    ["area"],
                    data=df_y,
                    shape_data=shape_data,
                    method=m,
                    verbose=False,
                )
                if w is not None:
                    mi = Moran(Y, w, transformation="R")
                    d[f"I_{m}"] = mi.I
                    d[f"p_{m}"] = mi.p_sim
                    d[f"VI_{m}"] = mi.VI_rand
            pv_inst_moran_results.append(d)
    pv_inst_moran_results = pd.DataFrame(pv_inst_moran_results)

    pv_cap_per_inst_moran_results = []
    for y in df["year"].unique():
        df_y = df[df["year"] == y].copy()
        if len(df_y) > 0 and df_y[variables[2]].notna().sum() > 0:
            d = {"year": y}
            for m in methods:
                w, X, Y = compute_weights(
                    variables[2],
                    ["area"],
                    data=df_y,
                    shape_data=shape_data,
                    method=m,
                    verbose=False,
                )
                if w is not None:
                    mi = Moran(Y, w, transformation="R")
                    d[f"I_{m}"] = mi.I
                    d[f"p_{m}"] = mi.p_sim
                    d[f"VI_{m}"] = mi.VI_rand
            pv_cap_per_inst_moran_results.append(d)
    pv_cap_per_inst_moran_results = pd.DataFrame(pv_cap_per_inst_moran_results)

    moran_results = {
        "pv_cap": pv_cap_moran_results,
        "pv_inst": pv_inst_moran_results,
        "pv_cap_per_inst": pv_cap_per_inst_moran_results,
    }
    with open(moran_results_file, "wb") as f:
        pickle.dump(moran_results, f)
else:
    with open(moran_results_file, "rb") as f:
        moran_results = pickle.load(f)
        pv_cap_moran_results, pv_inst_moran_results, pv_cap_per_inst_moran_results = (
            moran_results["pv_cap"],
            moran_results["pv_inst"],
            moran_results["pv_cap_per_inst"],
        )

labels = {
    # "log_pv_cap": "PV installed capacity",
    # "log_pv_inst": "PV installations",
    # "log_pv_cap_per_inst": "PV capacity per installation",
    "pv_cap": "PV installed capacity",
    "pv_inst": "PV installations",
    "pv_cap_per_inst": "PV capacity per installation",
    "I_inverse_distance": r"$W_{\text{inverse distance}}$",
    "I_k_nearest": r"$W_{\text{k-nearest}}$",
    "I_queen": r"$W_{\text{queen}}$",
}

fig = plt.figure(figsize=(6.0 * 3, 5.5))
ax_left = fig.add_subplot(1, 3, 1)
d = pv_cap_moran_results
for m in methods:
    i = d[["year"] + [i for i in d.columns if i.startswith(f"I_{m}")]].set_index("year")
    v = d[["year"] + [i for i in d.columns if i.startswith(f"VI_{m}")]].set_index(
        "year"
    )
    iv = average_morans_i(i, v)
    labels_ = {
        k: (
            f"{v}, ${iv['i_mean']:.3f}" + "^{" + f"{get_stars(iv['p-value'])}" + "}$"
        ).replace("*", r"{\ast}")
        for k, v in labels.items()
        if m in k
    }
    i = i.rename(columns=labels_)
    i.plot(
        ax=ax_left,
        marker="o",
    )
    ax_left.fill_between(
        i.index, i.iloc[:, 0] - v.iloc[:, 0], i.iloc[:, 0] + v.iloc[:, 0], alpha=0.25
    )
ax_left.set_title(labels[variables[0]], fontsize=11)
ax_left.set_xlabel(None)
ax_left.set_ylabel("Spatial autocorrelation measure (Moran's $I$)", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(1, 3, 2)
d = pv_inst_moran_results
for m in methods:
    i = d[["year"] + [i for i in d.columns if i.startswith(f"I_{m}")]].set_index("year")
    v = d[["year"] + [i for i in d.columns if i.startswith(f"VI_{m}")]].set_index(
        "year"
    )
    iv = average_morans_i(i, v)
    labels_ = {
        k: (
            f"{v}, ${iv['i_mean']:.3f}" + "^{" + f"{get_stars(iv['p-value'])}" + "}$"
        ).replace("*", r"{\ast}")
        for k, v in labels.items()
        if m in k
    }
    i = i.rename(columns=labels_)
    i.plot(
        ax=ax_middle,
        marker="o",
    )
    ax_middle.fill_between(
        i.index, i.iloc[:, 0] - v.iloc[:, 0], i.iloc[:, 0] + v.iloc[:, 0], alpha=0.25
    )
ax_middle.set_title(labels[variables[1]], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False
)

ax_right = fig.add_subplot(1, 3, 3)
d = pv_cap_per_inst_moran_results
for m in methods:
    i = d[["year"] + [i for i in d.columns if i.startswith(f"I_{m}")]].set_index("year")
    v = d[["year"] + [i for i in d.columns if i.startswith(f"VI_{m}")]].set_index(
        "year"
    )
    iv = average_morans_i(i, v)
    labels_ = {
        k: (
            f"{v}, ${iv['i_mean']:.3f}" + "^{" + f"{get_stars(iv['p-value'])}" + "}$"
        ).replace("*", r"{\ast}")
        for k, v in labels.items()
        if m in k
    }
    i = i.rename(columns=labels_)
    i.plot(
        ax=ax_right,
        marker="o",
    )
    ax_right.fill_between(
        i.index, i.iloc[:, 0] - v.iloc[:, 0], i.iloc[:, 0] + v.iloc[:, 0], alpha=0.25
    )
ax_right.set_title(labels[variables[2]], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False
)

fig.subplots_adjust(wspace=0.001)
fig.tight_layout(pad=1.01)
fig.savefig(
    os.path.join(results_dir, "fig_moran.pdf"),
    dpi=1200,
    bbox_inches="tight",
    format="pdf",
)
