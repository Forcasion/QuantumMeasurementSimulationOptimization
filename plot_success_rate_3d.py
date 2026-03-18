"""
plot_success_rate_3d.py

Reads files:
    output/summary_LD_SLSQP_nm1_ent{entanglement_target}_ops{num_ops}.csv
for entanglement_target in 1..5 and num_ops in 1..10,
then plots the last column (success rate) as a 3D bar chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Parameters ────────────────────────────────────────────────────────────────
ENT_RANGE = range(1, 6)    # entanglement_target: 1 → 5
OPS_RANGE = range(2, 11)   # num_ops:             2 → 10
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FILE_TEMPLATE = "summary_LD_SLSQP_nm1_ent{ent}_ops{ops}.csv"

# ── Collect data ──────────────────────────────────────────────────────────────
ent_vals = list(ENT_RANGE)
ops_vals = list(OPS_RANGE)

records = []
for ent in ent_vals:
    for ops in ops_vals:
        fpath = os.path.join(OUTPUT_DIR, FILE_TEMPLATE.format(ent=ent, ops=ops))
        df = pd.read_csv(fpath)
        value = float(df.iloc[:, -1].mean())
        records.append((ent, ops, value))

ent_arr = np.array([r[0] for r in records], dtype=float)
ops_arr = np.array([r[1] for r in records], dtype=float)
sr_arr  = np.array([r[2] for r in records], dtype=float)

# ── 3-D bar chart ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

bar_width  = 0.6
bar_depth  = 0.6

# Colour bars by success rate
colors = cm.viridis((sr_arr - sr_arr.min()) / (sr_arr.max() - sr_arr.min()))

ax.bar3d(
    ent_arr - bar_width / 2,   # x start
    ops_arr - bar_depth / 2,   # y start
    np.zeros_like(sr_arr),     # z start (floor)
    bar_width, bar_depth,      # dx, dy
    sr_arr,                    # dz = height
    color=colors,
    edgecolor="k", linewidth=0.3,
    alpha=1.0,
    zsort='average',
)

# Colourbar
mappable = cm.ScalarMappable(cmap=cm.viridis)
mappable.set_array(sr_arr)
fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1, label="Success rate")

ax.set_xlabel("Entanglement target", labelpad=10)
ax.set_ylabel("Num. operations", labelpad=10)
ax.set_zlabel("Success rate", labelpad=10)
ax.set_title("Success rate — 3-D bar chart", fontsize=12)
ax.set_xticks(ent_vals)
ax.set_yticks(ops_vals)

plt.tight_layout()
plt.savefig("success_rate_3d.png", dpi=150, bbox_inches="tight")
print("Plot saved to: success_rate_3d.png")
plt.show()