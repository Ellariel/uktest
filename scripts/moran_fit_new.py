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
from libpysal.weights import w_subset

from weights import compute_weights, compute_weights_new

warnings.simplefilter("ignore")

base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

print("data_dir:", data_dir)
print("results_dir:", results_dir)

df = pd.read_csv(os.path.join(data_dir, "df.csv"))
df = df[(df["year"] >= 2014) & (df["year"] <= 2019)]
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


def morans_i_time_permutation(
    X,
    W_master,
    n_perm=500,
    seed=13,
):
    """
    Global test for Moran's I across time using permutation.

    Parameters
    ----------
    X : ndarray (n_units, n_times)
        Data matrix with np.nan for missing observations.

    W_master : libpysal.weights.W
        Spatial weights for all units.

    n_perm : int
        Number of Monte Carlo permutations.

    seed : int or None

    Returns
    -------
    dict
    """

    rng = np.random.default_rng(seed)

    _, n_times = X.shape

    # ---------- observed ----------
    I_obs = np.empty(n_times)
    EI_obs = np.empty(n_times)
    VI_obs = np.empty(n_times)

    for t in range(n_times):
        keep = ~np.isnan(X[:, t])

        x = X[keep, t]

        ids = np.asarray(W_master.id_order)[keep]
        # Wt = W_master.subset(ids)
        Wt = w_subset(W_master, ids)
        print(
            len(ids),
            Wt.n,
            Wt.n_components,
            Wt.pct_nonzero,
        )

        mi = Moran(x, Wt, permutations=0)

        I_obs[t] = mi.I
        EI_obs[t] = mi.EI
        VI_obs[t] = mi.VI_norm

        print(
            t,
            len(x),
            mi.I,
            mi.EI,
            mi.VI_norm,
            # mi.s0,
            np.isnan(x).sum(),
        )

    z_obs = (I_obs - EI_obs) / np.sqrt(VI_obs)

    weights_obs = 1 / VI_obs
    weights_obs /= weights_obs.sum()

    T_obs = np.sum(weights_obs * z_obs)

    # ---------- permutation ----------
    T_perm = np.empty(n_perm)

    for p in range(n_perm):
        I_perm = np.empty(n_times)
        EI_perm = np.empty(n_times)
        VI_perm = np.empty(n_times)

        for t in range(n_times):
            keep = ~np.isnan(X[:, t])

            x = X[keep, t].copy()

            # permute available observations only
            x = rng.permutation(x)

            ids = np.asarray(W_master.id_order)[keep]
            # Wt = W_master.subset(ids)
            Wt = w_subset(W_master, ids)

            mi = Moran(x, Wt, permutations=0)

            I_perm[t] = mi.I
            EI_perm[t] = mi.EI
            VI_perm[t] = mi.VI_norm

        z_perm = (I_perm - EI_perm) / np.sqrt(VI_perm)

        # recompute weights for this permutation
        weights_perm = 1 / VI_perm
        weights_perm /= weights_perm.sum()

        T_perm[p] = np.sum(weights_perm * z_perm)

    p_value = (np.sum(np.abs(T_perm) >= np.abs(T_obs)) + 1) / (n_perm + 1)

    return {
        "I_time": I_obs,
        "z_time": z_obs,
        "weights": weights_obs,
        "statistic": T_obs,
        "p_value": p_value,
        "null_distribution": T_perm,
    }


# Global spatial autocorrelation (Moran’s I)
methods = ["inverse_distance", "k_nearest", "queen"]
variables = ["pv_cap_fit", "pv_inst_fit", "morans_i_time"]

moran_results_file = os.path.join(results_dir, "moran_results_fit.pickle")
if not os.path.exists(moran_results_file):
    """
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
                # print(df_y)
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
    """
    """
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
    """

    pv_cap_morans_i_time_results = {}
    for m in methods:
        w, X, Y = compute_weights_new(
            variables[0],
            ["area"],
            data=df,
            shape_data=shape_data,
            method=m,
            verbose=False,
        )
        # print(Y, Y.shape)
        X = (
            Y[["code", "year", variables[0]]].pivot(
                index="code", columns="year", values=variables[0]
            )
            # .reindex(w.id_order)
        )
        # print(X, X.shape)
        # print(w, w.id_order)
        mi = morans_i_time_permutation(X.reset_index(drop=True).to_numpy(), w)
        print(f"Method: {m}, Moran's I time permutation p-value: {mi}")
        break

    moran_results = {
        variables[0]: pv_cap_moran_results,
        variables[1]: pv_inst_moran_results,
        # variables[2]: pv_cap_per_inst_moran_results,
    }
    with open(moran_results_file, "wb") as f:
        pickle.dump(moran_results, f)
else:
    with open(moran_results_file, "rb") as f:
        moran_results = pickle.load(f)
        (
            pv_cap_moran_results,
            pv_inst_moran_results,
        ) = (  # , pv_cap_per_inst_moran_results
            moran_results[variables[0]],
            moran_results[variables[1]],
            # moran_results["pv_cap_per_inst_fit"],
        )

labels = {
    # "log_pv_cap": "PV installed capacity",
    # "log_pv_inst": "PV installations",
    # "log_pv_cap_per_inst": "PV capacity per installation",
    "pv_cap_fit": "PV installed capacity, FIT",
    "pv_inst_fit": "PV installations, FIT",
    "log_pv_cap_fit": "PV installed capacity, FIT",
    "log_pv_inst_fit": "PV installations, FIT",
    # "pv_cap_per_inst_fit": "PV capacity per installation",
    "I_inverse_distance": r"$W_{\text{inverse distance}}$",
    "I_k_nearest": r"$W_{\text{k-nearest}}$",
    "I_queen": r"$W_{\text{queen}}$",
}

fig = plt.figure(figsize=(6.0 * 2, 5.5))
ax_left = fig.add_subplot(1, 2, 1)
d = pv_cap_moran_results
for m in methods:
    i = d[["year"] + [i for i in d.columns if i.startswith(f"I_{m}")]].set_index("year")
    v = d[["year"] + [i for i in d.columns if i.startswith(f"VI_{m}")]].set_index(
        "year"
    )
    p = d[["year"] + [i for i in d.columns if i.startswith(f"p_{m}")]].set_index("year")
    # print(p)
    iv = {"i_mean": 5}  # average_morans_i(i, v)
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
    loc="upper right", fontsize=9, frameon=False
)  # bbox_to_anchor=(0.5, -0.05),

ax_middle = fig.add_subplot(1, 2, 2)
d = pv_inst_moran_results
for m in methods:
    i = d[["year"] + [i for i in d.columns if i.startswith(f"I_{m}")]].set_index("year")
    v = d[["year"] + [i for i in d.columns if i.startswith(f"VI_{m}")]].set_index(
        "year"
    )
    iv = {"i_mean": 5}  # average_morans_i(i, v)
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
    loc="upper right", fontsize=9, frameon=False
)  # bbox_to_anchor=(0.5, -0.05),
"""
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
"""
fig.subplots_adjust(wspace=0.001)
fig.tight_layout(pad=1.01)
fig.savefig(
    os.path.join(results_dir, "fig_moran_fit.pdf"),
    dpi=1200,
    bbox_inches="tight",
    format="pdf",
)
