from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

def plot_boot_barplot(boot_G_raw: Dict[str, Sequence[float]], tblG: pd.DataFrame, title: str = "Bootstrap PRISM-G") -> Tuple[plt.Figure, plt.Axes]:
    order = tblG.sort_values("mean_G", ascending=True)["name"].tolist()
    data  = [np.asarray(boot_G_raw[n], dtype=float) for n in order]

    means = np.array([d.mean() for d in data])
    sds   = np.array([d.std(ddof=1) for d in data])

    cmap = get_cmap("RdYlGn_r")      
    norm = Normalize(vmin=0, vmax=100)
    colors = [cmap(norm(m)) for m in means]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=sds, capsize=4)

    for bar, c in zip(bars, colors):
        bar.set_color(c)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("PRISM-G (0 = safer, 100 = riskier)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="PRISM-G Score")

    fig.tight_layout()
    return fig, ax

def plot_prismg_summary(df, title="PRISM-G (0 = safer, 100 = riskier)"):
    df_cand = df[df["role"] == "candidate"].sort_values("PRISM_G")

    labels = df_cand["name"].tolist()
    values = df_cand["PRISM_G"].tolist()

    cmap = get_cmap("RdYlGn_r")
    norm = Normalize(vmin=0, vmax=100)
    colors = [cmap(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(values))

    ax.bar(x, values, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("PRISM-G (0 = safer, 100 = riskier)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label="PRISM-G Score")

    fig.tight_layout()
    return fig, ax