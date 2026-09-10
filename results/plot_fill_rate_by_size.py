"""Gap fill rate by size bucket and direction.

Reads results/fill_rate_table.csv (written by scripts/analyze_gaps.py) and
plots the same_quarter_% column, the share of gaps filled within the same
quarter, as paired columns per size bucket. Writes results/fill_rate_by_size.png.

Standalone: needs only matplotlib. Run from anywhere:
    python results/plot_fill_rate_by_size.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

HERE = Path(__file__).resolve().parent
SRC = HERE / "fill_rate_table.csv"
OUT = HERE / "fill_rate_by_size.png"
COL = "same_quarter_%"

# Chart chrome: light-mode palette and a few small drawing helpers.
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASELINE = "#c3c2b7"
SERIES1 = "#2a78d6"; SERIES2 = "#eb6834"
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "hatch.linewidth": 0.9, "savefig.facecolor": SURFACE,
    })


def figure():
    # 1600 x 900 px at dpi 150
    return plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)


def px2data(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0)); x1, y1 = inv.transform((px, px))
    return x1 - x0, y1 - y0


def bar(ax, x, h, w, color, r_px=4, hatch=None, hatch_color=None, z=3):
    """Column <= 24px thick, 4px rounded data-end, square at the baseline."""
    if h == 0:
        return
    rx, ry = px2data(ax, r_px)
    y0, hh = (0, h) if h > 0 else (h, -h)
    kw = dict(facecolor=color, edgecolor=hatch_color if hatch else "none",
              linewidth=0, hatch=hatch, zorder=z)
    ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, hh,
                                boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, **kw))
    # square off the baseline end by covering its two rounded corners
    yb = 0 if h > 0 else -ry
    ax.add_patch(Rectangle((x - w / 2, yb), w, ry, **kw))


def strip(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE); ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


rows = list(csv.DictReader(open(SRC)))
buckets = []
for r in rows:
    if r["bucket"] not in buckets:
        buckets.append(r["bucket"])
data = {(r["direction"], r["bucket"]): r for r in rows}
n_total = sum(int(r["n_gaps"]) for r in rows)

setup(); fig = figure()
ax = fig.add_axes([0.06, 0.20, 0.92, 0.60])
ax.set_xlim(-0.5, len(buckets) - 0.5); ax.set_ylim(0, 100)
strip(ax)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
ax.set_xticks(range(len(buckets)))
ax.set_xticklabels([f"{b} gap" for b in buckets], fontsize=11, color=INK2)

wx, _ = px2data(ax, 24); gx, _ = px2data(ax, 26)  # 26px air between bars so each cap label clears its neighbour
series = [("up", "Up gap", SERIES1), ("down", "Down gap", SERIES2)]
table = []
for i, b in enumerate(buckets):
    for j, (d, label, color) in enumerate(series):
        r = data[(d, b)]; v = float(r[COL]); n = int(r["n_gaps"])
        x = i + (j - 0.5) * (wx + gx)
        bar(ax, x, v, wx, color)
        ax.text(x, v + 2.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK2)
        table.append((d, b, n, v))
    nu = int(data[("up", b)]["n_gaps"]); nd = int(data[("down", b)]["n_gaps"])
    ax.text(i, -9, f"n = {nu:,} up · {nd:,} down", ha="center", va="top", fontsize=8.5, color=MUTED)

ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in series], loc="upper right",
          frameon=False, fontsize=10, labelcolor=INK2, handlelength=1.0, handleheight=1.0, borderaxespad=0)

fig.text(0.06, 0.93, "Gap fill rate by size and direction", fontsize=17, fontweight="bold", color=INK, va="top")
fig.text(0.06, 0.875, f"503 S&P 500 tickers, 10 years of daily bars, {n_total:,} gap events",
         fontsize=11, color=INK2, va="top")
fig.text(0.06, 0.045, "Fill rate = share of gaps filled within the same quarter (same_quarter_% in fill_rate_table.csv). "
         "Gap size = open vs prior close, |gap| ≥ 1%.", fontsize=8.5, color=MUTED, va="bottom")
fig.savefig(OUT, dpi=150)
print("wrote", OUT, "n_total", n_total)
for t in table: print(t)
