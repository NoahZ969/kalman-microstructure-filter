"""Shared matplotlib style: thin marks, recessive grid, fixed categorical order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated CVD-safe for adjacent lines); never cycled.
C = {
    "raw": "#a8a7a2",       # neutral gray for the noisy observation
    "truth": "#0b0b0b",     # ink for ground truth
    "s1": "#2a78d6",        # blue
    "s2": "#eb6834",        # orange
    "s3": "#1baf7a",        # aqua
    "s4": "#eda100",        # yellow
}
TEXT2 = "#52514e"


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": "#c9c8c3",
        "axes.labelcolor": TEXT2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
