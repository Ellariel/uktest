import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# palette = sns.color_palette("tab20", 12)

palette = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#393b79",  # navy
    "#8dd3c7",  # mint
]


base_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(base_dir, ".."))
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

print("data_dir:", data_dir)
print("results_dir:", results_dir)

df = pd.read_csv(os.path.join(data_dir, "df.csv"))
df = df.sort_values(by=["year", "name"])

labels = {
    "log_pv_cap": "PV installed capacity [MW]",
    "log_pv_inst": "PV installations [Count]",
    "pv_cap_per_inst": "PV capacity per installation [MW / installation]",
}

fig = plt.figure(figsize=(6.0 * 3, 5.5))
ax_left = fig.add_subplot(1, 3, 1)
d = df.copy()
d = (
    d[["year", "pv_cap", "region"]]
    .groupby(["year", "region"])
    .mean()
    .reset_index()
    .pivot(index="year", columns="region", values="pv_cap")
)
d = d.rename(columns=labels)
d.plot(ax=ax_left, marker="o", color=palette, linewidth=2)
ax_left.set_title(labels["log_pv_cap"], fontsize=11)
ax_left.set_xlabel(None)
ax_left.set_ylabel(None, labelpad=-5)
ax_left.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False, ncols=3
)

ax_middle = fig.add_subplot(1, 3, 2)
d = df.copy()
d = (
    d[["year", "pv_inst", "region"]]
    .groupby(["year", "region"])
    .mean()
    .reset_index()
    .pivot(index="year", columns="region", values="pv_inst")
)
d = d.rename(columns=labels)
d.plot(ax=ax_middle, marker="o", color=palette, linewidth=2)
ax_middle.set_title(labels["log_pv_inst"], fontsize=11)
ax_middle.set_xlabel(None)
ax_middle.set_ylabel(None, labelpad=-5)
ax_middle.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False, ncols=3
)

ax_right = fig.add_subplot(1, 3, 3)
d = df.copy()
d = (
    d[["year", "pv_cap_per_inst", "region"]]
    .groupby(["year", "region"])
    .mean()
    .reset_index()
    .pivot(index="year", columns="region", values="pv_cap_per_inst")
)
d = d.rename(columns=labels)
d.plot(ax=ax_right, marker="o", logy=True, color=palette, linewidth=2)
ax_right.set_title(labels["pv_cap_per_inst"], fontsize=11)
ax_right.set_xlabel(None)
ax_right.set_ylabel(None, labelpad=-5)
ax_right.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False, ncols=3
)

fig.subplots_adjust(wspace=0.001)
fig.tight_layout(pad=1.01)
fig.savefig(
    os.path.join(results_dir, "fig_dyn.pdf"),
    dpi=1200,
    bbox_inches="tight",
    format="pdf",
)
