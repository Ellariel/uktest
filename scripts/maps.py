import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter


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
).rename(columns={"LAD24CD": "code", "LAD24NM": "name"})

df2014 = df[df["year"] == 2014].rename(
    columns={
        "pv_cap": "pv_cap_2014",
        "pv_inst": "pv_inst_2014",
        "pv_cap_per_inst": "pv_cap_per_inst_2014",
    }
)
df2022 = df[df["year"] == 2022].rename(
    columns={
        "pv_cap": "pv_cap_2022",
        "pv_inst": "pv_inst_2022",
        "pv_cap_per_inst": "pv_cap_per_inst_2022",
    }
)

df = shape_data.join(
    df2014[["code", "pv_cap_2014", "pv_inst_2014", "pv_cap_per_inst_2014"]].set_index(
        "code"
    ),
    on="code",
)
df = df.join(
    df2022[["code", "pv_cap_2022", "pv_inst_2022", "pv_cap_per_inst_2022"]].set_index(
        "code"
    ),
    on="code",
)

labels = {
    "pv_cap": "PV installed capacity\n[MW]",
    "pv_inst": "PV installations\n[Count]",
    "pv_cap_per_inst": "PV capacity per installation\n[MW / installation]",
}

vmin_cap = df[["pv_cap_2014", "pv_cap_2022"]].min().min()
vmax_cap = df[["pv_cap_2014", "pv_cap_2022"]].max().max()

vmin_inst = df[["pv_inst_2014", "pv_inst_2022"]].min().min()
vmax_inst = df[["pv_inst_2014", "pv_inst_2022"]].max().max()

vmin_ratio = df[["pv_cap_per_inst_2014", "pv_cap_per_inst_2022"]].min().min()
vmax_ratio = df[["pv_cap_per_inst_2014", "pv_cap_per_inst_2022"]].max().max()

legend_kwds = {"fraction": 0.02, "pad": 0.01}
cmap = plt.get_cmap("tab20b")
fig = plt.figure(figsize=(9.5, 6))

year = 2014

ax_left_top = fig.add_subplot(2, 3, 1)
df.plot(
    ax=ax_left_top,
    cmap=cmap,
    vmin=vmin_cap,
    vmax=vmax_cap,
    column=f"pv_cap_{year}",
    legend=True,
    legend_kwds=legend_kwds,
)
df.boundary.plot(ax=ax_left_top, color="darkgray", linewidth=0.06)
cbar_ax = ax_left_top.get_figure().axes[-1]  # last axis = colorbar
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar_ax.yaxis.set_major_formatter(fmt)
ax_left_top.set_title(labels["pv_cap"], fontsize=10)
ax_left_top.set_xlabel(None)
ax_left_top.set_ylabel(None)
ax_left_top.set_axis_off()
ax_left_top.text(
    0.05,
    0.95,
    "a) 2014",
    transform=ax_left_top.transAxes,
    ha="center",
    va="center",
    fontsize=9,
)

ax_middle_top = fig.add_subplot(2, 3, 2)
df.plot(
    ax=ax_middle_top,
    cmap=cmap,
    vmin=vmin_inst,
    vmax=vmax_inst,
    column=f"pv_inst_{year}",
    legend=True,
    legend_kwds=legend_kwds,
)
df.boundary.plot(ax=ax_middle_top, color="darkgray", linewidth=0.06)
cbar_ax = ax_middle_top.get_figure().axes[-1]  # last axis = colorbar
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar_ax.yaxis.set_major_formatter(fmt)
ax_middle_top.set_title(labels["pv_inst"], fontsize=10)
ax_middle_top.set_xlabel(None)
ax_middle_top.set_ylabel(None)
ax_middle_top.set_axis_off()

ax_right_top = fig.add_subplot(2, 3, 3)
df.plot(
    ax=ax_right_top,
    cmap=cmap,
    vmin=vmin_ratio,
    vmax=vmax_ratio,
    column=f"pv_cap_per_inst_{year}",
    legend=True,
    legend_kwds=legend_kwds,
)
df.boundary.plot(ax=ax_right_top, color="darkgray", linewidth=0.06)
cbar_ax = ax_right_top.get_figure().axes[-1]  # last axis = colorbar
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar_ax.yaxis.set_major_formatter(fmt)
ax_right_top.set_title(labels["pv_cap_per_inst"], fontsize=10)
ax_right_top.set_xlabel(None)
ax_right_top.set_ylabel(None)
ax_right_top.set_axis_off()


year = 2022


ax_left_bottom = fig.add_subplot(2, 3, 4)
df.plot(
    ax=ax_left_bottom,
    cmap=cmap,
    vmin=vmin_cap,
    vmax=vmax_cap,
    column=f"pv_cap_{year}",
    legend=True,
    legend_kwds=legend_kwds,
)
df.boundary.plot(ax=ax_left_bottom, color="darkgray", linewidth=0.06)
cbar_ax = ax_left_bottom.get_figure().axes[-1]  # last axis = colorbar
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar_ax.yaxis.set_major_formatter(fmt)
# ax_left_bottom.set_title(labels["pv_cap"], fontsize=11)
ax_left_bottom.set_xlabel(None)
ax_left_bottom.set_ylabel(None)
ax_left_bottom.set_axis_off()
ax_left_bottom.text(
    0.05,
    0.95,
    "b) 2022",
    transform=ax_left_bottom.transAxes,
    ha="center",
    va="center",
    fontsize=9,
)

ax_middle_bottom = fig.add_subplot(2, 3, 5)
df.plot(
    ax=ax_middle_bottom,
    cmap=cmap,
    vmin=vmin_inst,
    vmax=vmax_inst,
    column=f"pv_inst_{year}",
    legend=True,
    legend_kwds=legend_kwds,
)
df.boundary.plot(ax=ax_middle_bottom, color="darkgray", linewidth=0.06)
cbar_ax = ax_middle_bottom.get_figure().axes[-1]  # last axis = colorbar
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar_ax.yaxis.set_major_formatter(fmt)
# ax_middle_bottom.set_title(labels["pv_inst"], fontsize=11)
ax_middle_bottom.set_xlabel(None)
ax_middle_bottom.set_ylabel(None)
ax_middle_bottom.set_axis_off()

ax_right_bottom = fig.add_subplot(2, 3, 6)
df.plot(
    ax=ax_right_bottom,
    cmap=cmap,
    vmin=vmin_ratio,
    vmax=vmax_ratio,
    column=f"pv_cap_per_inst_{year}",
    legend=True,
    legend_kwds=legend_kwds,
)
df.boundary.plot(ax=ax_right_bottom, color="darkgray", linewidth=0.06)
cbar_ax = ax_right_bottom.get_figure().axes[-1]  # last axis = colorbar
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar_ax.yaxis.set_major_formatter(fmt)
# ax_right_bottom.set_title(labels["pv_cap_per_inst"], fontsize=11)
ax_right_bottom.set_xlabel(None)
ax_right_bottom.set_ylabel(None)
ax_right_bottom.set_axis_off()


fig.subplots_adjust(wspace=0.001)
fig.tight_layout(pad=1.01)
fig.savefig(
    os.path.join(results_dir, "fig_map.png"),
    dpi=300,
    bbox_inches="tight",
    format="png",
)
