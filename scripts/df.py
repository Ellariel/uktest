import os
import numpy as np
import pandas as pd


def log_transform(x):
    return np.sign(x) * np.log10(1 + np.abs(x))


base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")

print("data_dir:", data_dir)

fame = pd.read_pickle(os.path.join(base_dir, "fame_data", "melted.pkl"))
osn = pd.read_csv(os.path.join(base_dir, "raw_data", "df.csv"), sep=";")
fit = pd.read_csv(os.path.join(base_dir, "raw_data", "FIT", "fit.csv"))

df = pd.merge(osn, fame, how="left", on=["name", "year"])
df = pd.merge(df, fit, how="left", on=["name", "year"])
df["firms"] = df[[c for c in df.columns if "count_notna" in c]].max(axis=1)
df.drop(
    [
        "count_turnover",
        "count_notna_turnover",
        "mean_turnover",
        "count_profit",
        "count_notna_profit",
        "mean_profit",
        "mean_bootstrap_profit",
        "gini_bootstrap_profit",
        "count_employees",
        "count_notna_employees",
        "mean_employees",
        "count_fixed_assets",
        "count_notna_fixed_assets",
        "mean_fixed_assets",
        "count_current_assets",
        "count_notna_current_assets",
        "mean_current_assets",
        "mean_bootstrap_turnover",
        "gini_bootstrap_turnover",
        "mean_bootstrap_employees",
        "gini_bootstrap_employees",
        "gini_bootstrap_fixed_assets",
        "gini_bootstrap_current_assets",
        "earn_gini",
        "gdp",
        "vad",
        "vad_tax",
        "w_on_inst",
        "w_on_cap",
    ],
    inplace=True,
    axis=1,
)
df.rename(
    columns={
        # "mean_bootstrap_turnover": "turnover",
        # "gini_bootstrap_turnover": "gini_turnover",
        # "mean_bootstrap_employees": "employees",
        # "gini_bootstrap_employees": "gini_employees",
        "mean_bootstrap_fixed_assets": "fixed_assets",
        # "gini_bootstrap_fixed_assets": "gini_fixed_assets",
        "mean_bootstrap_current_assets": "current_assets",
        # "gini_bootstrap_current_assets": "gini_current_assets",
        # "earn_gini": "gini_earnings",
        # "count": "firms",
        # "sunny_days": "irradiance",
    },
    inplace=True,
)

df["pv_cap"] = df["pv_cap"] * 1000  # MW -> kW
df["pv_cap_per_inst"] = df["pv_cap"] / df["pv_inst"]


df["pv_cap_fit"] = df["pv_cap_fit_cum"]
df["pv_inst_fit"] = df["pv_inst_fit_cum"]
df["pv_cap_per_inst_fit"] = df["pv_cap_fit"] / df["pv_inst_fit"]

df = df.sort_values(by=["name", "year"])
# df["pv_cap_diff"] = df["pv_cap"] - df["pv_cap"].shift(1)
# df["pv_inst_diff"] = df["pv_inst"] - df["pv_inst"].shift(1)
# df["pv_cap_per_inst_diff"] = df["pv_cap_diff"] / df["pv_inst_diff"]

df["fdensity"] = df["firms"] / df["area"]
df["pdensity"] = df["population"] / df["area"]
df["hdensity"] = df["hholds"] / df["area"]
df["assets"] = df["current_assets"] + df["fixed_assets"]
df["adensity"] = df["assets"] / df["area"]

df["fcapacity"] = df["firms"] / df["population"]
df["hcapacity"] = df["hholds"] / df["population"]
df["acapacity"] = df["assets"] / df["population"]

"""
def gini_total_weighted(gini_fixed, gini_current, mean_fixed, mean_current):
    mu_total = mean_fixed + mean_current
    return (mean_fixed / mu_total) * gini_fixed + (
        mean_current / mu_total
    ) * gini_current


df["gini_assets"] = gini_total_weighted(
    df["gini_fixed_assets"],
    df["gini_current_assets"],
    df["fixed_assets"],
    df["current_assets"],
)
"""

df.drop(
    ["fixed_assets", "current_assets", "pv_cap_fit_cum", "pv_inst_fit_cum"],
    inplace=True,
    axis=1,
)

for c in [
    # "turnover",
    # "employees",
    # "fixed_assets",
    # "current_assets",
    "assets",
    # "gdp",
    "income",
    # "vad",
    "gdp_cap",
    # "vad_tax",
    "pv_cap",
    "pv_inst",
    "pv_cap_per_inst",
    "pv_cap_fit",
    "pv_inst_fit",
    "pv_cap_per_inst_fit",
    # "pv_cap_diff",
    # "pv_inst_diff",
    # "pv_cap_per_inst_diff",
    # "w_on_inst",
    # "w_on_cap",
    "hholds",
    # "area",
    # "ddays",
    "earnings",
    "firms",
    "population",
    "house_price",
    # "irradiance",
    "fcapacity",
    "hcapacity",
    "acapacity",
    "sunny_hours",
]:
    if c in df.columns:
        df[f"log_{c}"] = log_transform(df[c])

df.to_csv(os.path.join(data_dir, "df.csv"), index=False)

print("N =", len(df.dropna()["name"].drop_duplicates()))
print("T =", len(df.dropna()["year"].dropna().drop_duplicates()))


df = pd.read_csv(os.path.join(data_dir, "df.csv"))
df = df.sort_values(by=["year", "name"])

d = df[["year", "name", "pv_cap", "pv_inst", "pv_cap_fit", "pv_inst_fit"]]
diffs = []
for _, data in d.groupby("name"):
    data = data.sort_values(by=["year"]).copy()
    data.set_index(["year", "name"], inplace=True)
    for i in data.columns:
        data[i] = pd.to_numeric(data[i])
        data[f"{i}_diff"] = data[i] - data[i].shift(1)
        data[f"log_{i}_diff"] = log_transform(data[f"{i}_diff"])
    data = data[[c for c in data.columns if "_diff" in c]]
    diffs.append(data.reset_index())

df = pd.merge(df, pd.concat(diffs, axis=0), how="left", on=["year", "name"])
df.to_csv(os.path.join(data_dir, "df.csv"), index=False)
