import numpy as np
import pandas as pd


def log_transform(x):
    return np.sign(x) * np.log10(1 + np.abs(x))


fame = pd.read_pickle("../fame_data/melted.pkl")
# fame = fame.loc[:,~fame.columns.duplicated()].copy()
osn = pd.read_csv("../raw_data/df.csv", sep=";")
df = pd.merge(osn, fame, how="left", on=["name", "year"])
df["count"] = df[[c for c in df.columns if "count_notna" in c]].max(axis=1)
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
    ],
    inplace=True,
    axis=1,
)
df.rename(
    columns={
        "mean_bootstrap_turnover": "turnover",
        "gini_bootstrap_turnover": "gini_turnover",
        "mean_bootstrap_employees": "employees",
        "gini_bootstrap_employees": "gini_employees",
        "mean_bootstrap_fixed_assets": "fixed_assets",
        "gini_bootstrap_fixed_assets": "gini_fixed_assets",
        "mean_bootstrap_current_assets": "current_assets",
        "gini_bootstrap_current_assets": "gini_current_assets",
        "earn_gini": "gini_earnings",
    },
    inplace=True,
)

# df["pv_inst_per_hholds"] = df["pv_inst"] / df["hholds"]
# df["pv_cap_per_hholds"] = df["pv_cap"] / df["hholds"]
df["pv_cap_per_inst"] = df["pv_cap"] / df["pv_inst"]
df["fdensity"] = df["count"] / df["area"]
df["pdensity"] = df["population"] / df["area"]
df["hdensity"] = df["hholds"] / df["area"]
df["assets"] = df["current_assets"] + df["fixed_assets"]
df["adensity"] = df["assets"] / df["area"]


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

for c in [
    "turnover",
    "employees",
    "fixed_assets",
    "current_assets",
    "assets",
    "gdp",
    "income",
    "vad",
    "gdp_cap",
    "vad_tax",
    "pv_cap",
    "pv_inst",
    "w_on_inst",
    "w_on_cap",
    "hholds",
    "area",
    "ddays",
    "earnings",
    "count",
    "population",
    "house_price",
    # "mean_irradiance",
    # "pv_inst_per_hholds",
    # "pv_cap_per_hholds",
    "pv_cap_per_inst",
]:
    if c in df.columns:
        df[f"log_{c}"] = log_transform(df[c])

df.to_csv("df.csv", index=False)

print("N =", len(df.dropna()["name"].drop_duplicates()))
print("T =", len(df.dropna()["year"].dropna().drop_duplicates()))
