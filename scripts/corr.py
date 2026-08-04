import os
import numpy as np
import pandas as pd
from numbers import Number
from scipy.stats import spearmanr, norm
import matplotlib.pyplot as plt
import seaborn as sns


base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

print("data_dir:", data_dir)
print("results_dir:", results_dir)

df = pd.read_csv(os.path.join(data_dir, "df.csv"))
df = df[(df["year"] >= 2014) & (df["year"] <= 2022)]
df = df.sort_values(by=["year", "name"])


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


def average_correlations(r, n, alpha=0.05, method="two-tailed"):
    """
    r : list/array of correlations
    n : list/array of sample sizes
    """
    r = np.array(r, dtype=float)
    n = np.array(n, dtype=float)

    z = np.arctanh(r)  # 1) Fisher z-transform
    w = n - 3  # 2) Weights (inverse variance)
    z_bar = np.sum(w * z) / np.sum(w)  # 3) Pooled effect (weighted mean z)
    se = 1 / np.sqrt(np.sum(w))  # 4) Standard error of pooled effect
    z = z_bar / se  # 5) Test statistic
    p = 1 - norm.cdf(np.abs(z))  # 6) Two-tailed p-value
    p = p * 2 if method == "two-tailed" else p
    r_bar = np.tanh(z_bar)  # 7) Convert back to correlation
    # 8) CI in z-space, then back-transform
    # z-critical = stats.norm.ppf(1 - alpha) (use alpha = alpha/2 for two-sided)
    a = alpha / 2 if method == "two-tailed" else alpha
    z_crit = norm.ppf(1 - a)
    z_ci_low = z_bar - z_crit * se
    z_ci_high = z_bar + z_crit * se
    r_ci_low = np.tanh(z_ci_low)
    r_ci_high = np.tanh(z_ci_high)
    return {
        "r_mean": r_bar,
        "z_mean": z_bar,
        "se": se,
        "z": z,
        "p-value": p,
        "cil": r_ci_low,
        "cir": r_ci_high,
    }


vars = [
    "log_gdp_cap",
    "log_income",
    "log_earnings",
    "log_population",
    "log_hholds",
    "log_firms",
    # "log_employees",
    "log_house_price",
    "pdensity",
    "hdensity",
    "fdensity",
    "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
    "log_fcapacity",
    "log_hcapacity",
    "log_acapacity",
]

vars = [
    # "gdp_cap",
    "income",
    # "earnings",
    "population",
    "hholds",
    "firms",
    # "log_employees",
    "house_price",
    "pdensity",
    "hdensity",
    "fdensity",
    "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
    "fcapacity",
    # "hcapacity",
    "acapacity",
]

corr_results = []
for k in ["pv_cap", "pv_inst", "pv_cap_fit", "pv_inst_fit"]:  # , "pv_cap_per_inst"]:
    d = {}
    for v in vars:
        df_v = df[[k, v, "year"]].dropna()
        if len(df_v) > 0:
            r, n = [], []
            for y in df_v["year"].unique():
                df_y = df_v[df_v["year"] == y].copy()
                if len(df_y) > 0 and df_y[k].notna().sum() > 0:
                    r_v, p = spearmanr(df_y[k], df_y[v], nan_policy="omit")
                    r.append(r_v)
                    n.append(len(df_y))
            r_a = average_correlations(r, n)
            d[v] = f"{r_a['r_mean']:.2f}" + "^{" + f"{get_stars(r_a['p-value'])}" + "}"
    corr_results.append({k: d})

corr_results = pd.concat(
    [
        pd.concat(
            [
                pd.DataFrame.from_dict(v, orient="index", columns=[k])
                for k, v in i.items()
            ],
            axis=0,
        )
        for i in corr_results
    ],
    axis=1,
)

pv_cap_corr_results = []
for y in df["year"].unique():
    df_y = df[df["year"] == y].copy()
    if len(df_y) > 0 and df_y["pv_cap"].notna().sum() > 0:
        d = {"year": y}
        for v in vars:
            r_v, p = spearmanr(df_y["pv_cap"], df_y[v], nan_policy="omit")
            d[f"r_{v}"] = r_v
            d[f"p_{v}"] = p
        pv_cap_corr_results.append(d)
pv_cap_corr_results = pd.DataFrame(pv_cap_corr_results)

pv_inst_corr_results = []
for y in df["year"].unique():
    df_y = df[df["year"] == y].copy()
    if len(df_y) > 0 and df_y["pv_inst"].notna().sum() > 0:
        d = {"year": y}
        for v in vars:
            r_v, p = spearmanr(df_y["pv_inst"], df_y[v], nan_policy="omit")
            d[f"r_{v}"] = r_v
            d[f"p_{v}"] = p
        pv_inst_corr_results.append(d)
pv_inst_corr_results = pd.DataFrame(pv_inst_corr_results)

pv_cap_fit_corr_results = []
for y in df["year"].unique():
    df_y = df[df["year"] == y].copy()
    if len(df_y) > 0 and df_y["pv_cap_fit"].notna().sum() > 0:
        d = {"year": y}
        for v in vars:
            r_v, p = spearmanr(df_y["pv_cap_fit"], df_y[v], nan_policy="omit")
            d[f"r_{v}"] = r_v
            d[f"p_{v}"] = p
        pv_cap_fit_corr_results.append(d)
pv_cap_fit_corr_results = pd.DataFrame(pv_cap_fit_corr_results)

pv_inst_fit_corr_results = []
for y in df["year"].unique():
    df_y = df[df["year"] == y].copy()
    if len(df_y) > 0 and df_y["pv_inst_fit"].notna().sum() > 0:
        d = {"year": y}
        for v in vars:
            r_v, p = spearmanr(df_y["pv_inst_fit"], df_y[v], nan_policy="omit")
            d[f"r_{v}"] = r_v
            d[f"p_{v}"] = p
        pv_inst_fit_corr_results.append(d)
pv_inst_fit_corr_results = pd.DataFrame(pv_inst_fit_corr_results)


"""
pv_cap_per_inst_corr_results = []
for y in df["year"].unique():
    df_y = df[df["year"] == y].copy()
    if len(df_y) > 0 and df_y["pv_cap_per_inst"].notna().sum() > 0:
        d = {"year": y}
        for v in vars:
            r_v, p = spearmanr(df_y["pv_cap_per_inst"], df_y[v], nan_policy="omit")
            d[f"r_{v}"] = r_v
            d[f"p_{v}"] = p
        pv_cap_per_inst_corr_results.append(d)
pv_cap_per_inst_corr_results = pd.DataFrame(pv_cap_per_inst_corr_results)
"""

labels = {  # "r_log_gdp_cap": "GDP per capita",
    "r_income": "Income",
    # "r_earnings": "Earnings",
    "r_house_price": "House price",
    "r_population": "Population",
    "r_hholds": "Households",
    "r_firms": "Firms",
    # "r_log_employees": "Employees",
    "r_pdensity": "$D$(Population)",
    "r_hdensity": "$D$(Households)",
    "r_fdensity": "$D$(Firms)",
    "r_adensity": "$D$(Firm assets)",
    # "r_gdp_cap": "GDP per capita",
    "r_fcapacity": "Firms per capita",
    # "r_hcapacity": "Households per capita",
    "r_acapacity": "Firm assets per capita",
    # "r_gini_earnings": "$G$(Earnings)",
    # "r_gini_assets": "$G$(Firm assets)",
    # "r_gini_employees": "$G$(Employees)",
    "pv_cap": "PV installed capacity, ONS",
    "pv_inst": "PV installations, ONS",
    "pv_cap_fit": "PV installed capacity, FIT",
    "pv_inst_fit": "PV installations, FIT",
    # "pv_cap_per_inst": "PV capacity per installation",
}

fig = plt.figure(figsize=(5.3 * 4, 4.5 * 3))
#############################ONS
vars1 = [  # "log_gdp_cap",
    "income",
    "house_price",
    # "earnings",
    "population",
    "hholds",
    # "log_employees",
    "firms",
    # "pdensity",
    # "hdensity",
    # "fdensity",
    # "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
]

ax_left = fig.add_subplot(3, 4, 1)
d = pv_cap_corr_results.copy()
d = d[["year"] + [c for v in vars1 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_left,
    marker="o",
)
ax_left.set_title(labels["pv_cap"], fontsize=11)
ax_left.set_xlabel(None)
# ax_left.set_ylabel("Spearman's correlation coefficient", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(3, 4, 2)
d = pv_inst_corr_results.copy()
d = d[["year"] + [c for v in vars1 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_middle,
    marker="o",
)
ax_middle.set_title(labels["pv_inst"], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""
ax_right = fig.add_subplot(3, 3, 3)
d = pv_cap_per_inst_corr_results.copy()
d = d[["year"] + [c for v in vars1 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_per_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_right,
    marker="o",
)
ax_right.set_title(labels["pv_cap_per_inst"], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""
vars2 = [  # "log_gdp_cap",
    # "log_income",
    # "log_earnings",
    # "log_population",
    # "log_hholds",
    # "log_employees",
    "pdensity",
    "hdensity",
    "fdensity",
    "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
]

ax_left = fig.add_subplot(3, 4, 5)
d = pv_cap_corr_results.copy()
d = d[["year"] + [c for v in vars2 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_left,
    marker="o",
)
# ax_left.set_title(labels['log_pv_cap'], fontsize=11)
ax_left.set_xlabel(None)
ax_left.set_ylabel("Spearman's correlation coefficient", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(3, 4, 6)
d = pv_inst_corr_results.copy()
d = d[["year"] + [c for v in vars2 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_middle,
    marker="o",
)
# ax_middle.set_title(labels['log_pv_inst'], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""
ax_right = fig.add_subplot(3, 2, 5)
d = pv_cap_per_inst_corr_results.copy()
d = d[["year"] + [c for v in vars2 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_per_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_right,
    marker="o",
)
# ax_right.set_title(labels['pv_cap_per_inst'], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""

vars3 = [  # "log_gdp_cap",
    # "log_income",
    # "log_earnings",
    # "log_population",
    # "log_hholds",
    # "log_employees",
    # "pdensity",
    # "hdensity",
    # "fdensity",
    # "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
    # "gdp_cap",
    "fcapacity",
    # "hcapacity",
    "acapacity",
]

ax_left = fig.add_subplot(3, 4, 9)
d = pv_cap_corr_results.copy()
d = d[["year"] + [c for v in vars3 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_left,
    marker="o",
)
# ax_left.set_title(labels['log_pv_cap'], fontsize=11)
ax_left.set_xlabel(None)
# ax_left.set_ylabel("Spearman's correlation coefficient", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(3, 4, 10)
d = pv_inst_corr_results.copy()
d = d[["year"] + [c for v in vars3 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_middle,
    marker="o",
)
# ax_middle.set_title(labels['log_pv_inst'], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=9, frameon=False
)
"""
ax_right = fig.add_subplot(3, 3, 9)
d = pv_cap_per_inst_corr_results.copy()
d = d[["year"] + [c for v in vars3 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_per_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_right,
    marker="o",
)
# ax_right.set_title(labels['pv_cap_per_inst'], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=9, frameon=False
)
"""

#############################FIT


vars1 = [  # "log_gdp_cap",
    "income",
    "house_price",
    # "earnings",
    "population",
    "hholds",
    # "log_employees",
    "firms",
    # "pdensity",
    # "hdensity",
    # "fdensity",
    # "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
]

ax_left = fig.add_subplot(3, 4, 3)
d = pv_cap_corr_results.copy()
d = d[["year"] + [c for v in vars1 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_fit']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_left,
    marker="o",
)
ax_left.set_title(labels["pv_cap_fit"], fontsize=11)
ax_left.set_xlabel(None)
# ax_left.set_ylabel("Spearman's correlation coefficient", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(3, 4, 4)
d = pv_inst_corr_results.copy()
d = d[["year"] + [c for v in vars1 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_inst_fit']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_middle,
    marker="o",
)
ax_middle.set_title(labels["pv_inst_fit"], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""
ax_right = fig.add_subplot(3, 3, 3)
d = pv_cap_per_inst_corr_results.copy()
d = d[["year"] + [c for v in vars1 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_per_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_right,
    marker="o",
)
ax_right.set_title(labels["pv_cap_per_inst"], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""
vars2 = [  # "log_gdp_cap",
    # "log_income",
    # "log_earnings",
    # "log_population",
    # "log_hholds",
    # "log_employees",
    "pdensity",
    "hdensity",
    "fdensity",
    "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
]

ax_left = fig.add_subplot(3, 4, 7)
d = pv_cap_corr_results.copy()
d = d[["year"] + [c for v in vars2 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_fit']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_left,
    marker="o",
)
# ax_left.set_title(labels['log_pv_cap'], fontsize=11)
ax_left.set_xlabel(None)
ax_left.set_ylabel("Spearman's correlation coefficient", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(3, 4, 8)
d = pv_inst_corr_results.copy()
d = d[["year"] + [c for v in vars2 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_inst_fit']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_middle,
    marker="o",
)
# ax_middle.set_title(labels['log_pv_inst'], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""
ax_right = fig.add_subplot(3, 2, 5)
d = pv_cap_per_inst_corr_results.copy()
d = d[["year"] + [c for v in vars2 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_per_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_right,
    marker="o",
)
# ax_right.set_title(labels['pv_cap_per_inst'], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
)
"""

vars3 = [  # "log_gdp_cap",
    # "log_income",
    # "log_earnings",
    # "log_population",
    # "log_hholds",
    # "log_employees",
    # "pdensity",
    # "hdensity",
    # "fdensity",
    # "adensity",
    # "gini_assets",
    # "gini_earnings",
    # "gini_employees",
    # "gdp_cap",
    "fcapacity",
    # "hcapacity",
    "acapacity",
]

ax_left = fig.add_subplot(3, 4, 11)
d = pv_cap_corr_results.copy()
d = d[["year"] + [c for v in vars3 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_fit']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_left,
    marker="o",
)
# ax_left.set_title(labels['log_pv_cap'], fontsize=11)
ax_left.set_xlabel(None)
# ax_left.set_ylabel("Spearman's correlation coefficient", fontsize=11)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=9, frameon=False
)

ax_middle = fig.add_subplot(3, 4, 12)
d = pv_inst_corr_results.copy()
d = d[["year"] + [c for v in vars3 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_inst_fit']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_middle,
    marker="o",
)
# ax_middle.set_title(labels['log_pv_inst'], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel("", labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=9, frameon=False
)
"""
ax_right = fig.add_subplot(3, 3, 9)
d = pv_cap_per_inst_corr_results.copy()
d = d[["year"] + [c for v in vars3 for c in d.columns if v in str(c)]]
d = d[["year"] + [i for i in d.columns if i.startswith("r_")]]
labels_ = {
    k: f"{v}, ${r['pv_cap_per_inst']}$".replace("*", r"{\ast}")
    for k, v in labels.items()
    for i, r in corr_results.iterrows()
    if i == k[2:]
}
d = d.rename(columns=labels_).set_index("year")
d.plot(
    ax=ax_right,
    marker="o",
)
# ax_right.set_title(labels['pv_cap_per_inst'], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel("", labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, fontsize=9, frameon=False
)
"""


fig.subplots_adjust(wspace=0.001)
fig.tight_layout(pad=1.01)
fig.savefig(
    os.path.join(results_dir, "fig_corr1.png"),
    dpi=1200,
    bbox_inches="tight",
    format="png",
)


######FIG2


vars = [
    "pv_cap",
    "pv_inst",
    "pv_cap_fit",
    "pv_inst_fit",
    # "pv_cap_per_inst",
    # "log_gdp_cap",
    "house_price",
    "income",
    # "earnings",
    "assets",
    "population",
    "hholds",
    "firms",
    # "log_employees",
    "pdensity",
    "hdensity",
    "fdensity",
    "adensity",
    # "gini_earnings",
    # "gini_assets",
    # "gini_employees",
    # "gdp_cap",
    "fcapacity",
    # "hcapacity",
    "acapacity",
]
full_corr_results_r = {}
full_corr_results_p = {}
for v1 in vars:
    full_corr_results_r.setdefault(v1, {})
    full_corr_results_p.setdefault(v1, {})
    for v2 in vars:
        if v1 == v2:
            continue
        full_corr_results_r[v1].setdefault(v2, 0)
        full_corr_results_p[v1].setdefault(v2, 0)
        df_v = df[[v1, v2, "year"]].dropna()
        if len(df_v) > 0:
            r, n = [], []
            for y in df_v["year"].unique():
                df_y = df_v[df_v["year"] == y].copy()
                if len(df_y) > 0 and df_y[v1].notna().sum() > 0:
                    r_v, p = spearmanr(df_y[v1], df_y[v2], nan_policy="omit")
                    r.append(r_v)
                    n.append(len(df_y))
            r_a = average_correlations(r, n)
            full_corr_results_r[v1][v2] = r_a[
                "r_mean"
            ]  # f"{r_a["r_mean"]:.2f}" + "^{" + f"{get_stars(r_a['p-value'])}" + "}"
            full_corr_results_p[v1][v2] = r_a["p-value"]

labels = {
    "pv_cap": "PV inst.\ncapacity, ONS",
    "pv_inst": "PV inst., ONS",
    "pv_cap_fit": "PV inst.\ncapacity, FIT",
    "pv_inst_fit": "PV inst., FIT",
    # "pv_cap_per_inst": "PV capacity\nper inst.",
    # "log_gdp_cap": "GDP\nper capita\n",
    "income": "Income",
    # "earnings": "Earnings",
    "assets": "Firm assets",
    "house_price": "House price",
    "population": "Population",
    "hholds": "Households",
    "firms": "Firms",
    # "log_employees": "Employees",
    "pdensity": "$D$(Population)",
    "hdensity": "$D$(Households)",
    "fdensity": "$D$(Firms)",
    "adensity": "$D$(Firm assets)",
    # "gini_earnings": "$G$(Earnings)",
    # "gini_assets": "$G$(Firm assets)",
    # "gini_employees": "$G$(Employees)",
    # "gdp_cap": "GDP\nper capita\n",
    "fcapacity": "Firms\nper capita",
    # "hcapacity": "Households\nper capita",
    "acapacity": "Firm assets\nper capita",
}

fig = plt.figure(figsize=(12.5, 9))
ax_left = fig.add_subplot()
c = pd.DataFrame(full_corr_results_r).copy()
c = c.reindex(index=labels.keys(), columns=labels.keys())
p = pd.DataFrame(full_corr_results_p).copy()
p = p.reindex(index=labels.keys(), columns=labels.keys())
a = c.copy().astype(str)
for i in a.keys():
    for j in a[i].keys():
        a.loc[i, j] = f"${c[i][j]:.2f}^{{{get_stars(p[i][j])}}}$".replace(
            "*", r"{\ast}"
        )
c = c.rename(columns=labels)
c.index = c.index.map(labels)
cmap = sns.diverging_palette(230, 20, n=256, as_cmap=True)  # cmap="YlGnBu"
mask = np.triu(np.ones_like(c, dtype=bool))
ax_left = sns.heatmap(
    c,
    cmap=cmap,
    mask=mask,
    annot_kws={"va": "center", "fontsize": 7.8},
    fmt="",
    robust=False,
    ax=ax_left,
    cbar=True,
    annot=a,
)
fig.tight_layout(pad=1.01)
fig.savefig(
    os.path.join(results_dir, "fig_corr2.png"),
    dpi=1200,
    bbox_inches="tight",
    format="png",
)
