"""
Problem 1: Formation control for N=20 agents displaying the name KUSHAL
letter by letter in R^2, over a fixed connected Erdos-Renyi communication graph.
"""

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

from scipy.optimize import linear_sum_assignment
import matplotlib.animation as animation


# ===============================================================
# 1. INITIAL SETTINGS
# ===============================================================

rng = np.random.default_rng(2025)

N = 20
NAME = "KUSHAL"

p = 0.25
seed = 0


# ===============================================================
# 2. COMMUNICATION GRAPH
# ===============================================================

G = nx.erdos_renyi_graph(
    N,
    p,
    seed=seed
)

# Regenerate until the graph is connected
while not nx.is_connected(G):
    seed += 1
    G = nx.erdos_renyi_graph(
        N,
        p,
        seed=seed
    )

L = nx.laplacian_matrix(G).toarray().astype(float)

print(
    f"Graph: N={N}, p={p}, "
    f"edges={G.number_of_edges()}, "
    f"connected={nx.is_connected(G)}"
)


# Save communication graph
plt.figure(figsize=(5, 5))

nx.draw(
    G,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=350,
    font_size=8,
    font_weight="bold"
)

plt.title(
    f"Communication graph $G(V,E)$, N={N}, p={p}"
)

plt.tight_layout()

plt.savefig(
    "comm_graph.png",
    dpi=200
)

plt.close()

print("Saved comm_graph.png")


# ===============================================================
# 3. CREATE POINTS FOR EACH LETTER
# ===============================================================

def letter_points(letter, n_points, spread_scale=1.0):

    """
    Sample n_points evenly along the OUTLINE of the glyph.
    This allows the agents to form the shape of each letter.
    """

    fp = FontProperties(
        family="DejaVu Sans",
        weight="bold"
    )

    tp = TextPath(
        (0, 0),
        letter,
        size=10,
        prop=fp
    )

    poly_paths = tp.to_polygons()

    # If no polygon is obtained
    if len(poly_paths) == 0:

        return rng.uniform(
            -1,
            1,
            size=(n_points, 2)
        )

    # -----------------------------------------------------------
    # Calculate arc length of every polygon
    # -----------------------------------------------------------

    seg_lens = []

    for poly in poly_paths:

        poly = np.array(poly)

        d = np.linalg.norm(
            np.diff(
                poly,
                axis=0,
                append=poly[:1]
            ),
            axis=1
        )

        seg_lens.append(d.sum())

    total_len = sum(seg_lens)

    pts = []

    remaining = n_points

    # -----------------------------------------------------------
    # Sample points along each polygon
    # -----------------------------------------------------------

    for k, poly in enumerate(poly_paths):

        poly = np.array(poly)

        if k < len(poly_paths) - 1:

            n_here = max(
                1,
                round(
                    n_points *
                    seg_lens[k] /
                    total_len
                )
            )

        else:

            n_here = remaining

        n_here = min(
            n_here,
            remaining
        )

        d = np.linalg.norm(
            np.diff(
                poly,
                axis=0,
                append=poly[:1]
            ),
            axis=1
        )

        cum = np.concatenate(
            [[0], np.cumsum(d)]
        )

        perimeter = cum[-1]

        if perimeter == 0 or n_here <= 0:
            continue

        s_samples = np.linspace(
            0,
            perimeter,
            n_here,
            endpoint=False
        )

        for s in s_samples:

            seg = (
                np.searchsorted(
                    cum,
                    s,
                    side="right"
                ) - 1
            )

            seg = min(
                seg,
                len(poly) - 1
            )

            t = (
                (s - cum[seg])
                /
                (d[seg] + 1e-12)
            )

            p1 = poly[seg]

            p2 = poly[
                (seg + 1) % len(poly)
            ]

            point = (
                p1 +
                t * (p2 - p1)
            )

            pts.append(point)

        remaining -= n_here

    pts = np.array(pts)

    # Make sure exactly n_points are returned
    if len(pts) < n_points:

        extra = pts[
            rng.integers(
                0,
                len(pts),
                size=n_points - len(pts)
            )
        ]

        pts = np.vstack(
            [pts, extra]
        )

    elif len(pts) > n_points:

        keep = rng.choice(
            len(pts),
            size=n_points,
            replace=False
        )

        pts = pts[keep]

    return pts * spread_scale


# ===============================================================
# 4. GENERATE TARGET POINTS FOR KUSHAL
# ===============================================================

letters = list(NAME)

n_letters = len(letters)

letter_width = 14.0

targets_per_letter = []


for i, ch in enumerate(letters):

    pts = letter_points(
        ch,
        N
    )

    pts[:, 0] += i * letter_width

    targets_per_letter.append(
        pts
    )


# ---------------------------------------------------------------
# Center each letter around the origin
# ---------------------------------------------------------------

targets_per_letter_centered = []


for ch in letters:

    pts = letter_points(
        ch,
        N
    )

    pts -= pts.mean(
        axis=0
    )

    targets_per_letter_centered.append(
        pts
    )


targets_per_letter = (
    targets_per_letter_centered
)


# ===============================================================
# 5. HUNGARIAN ASSIGNMENT
# ===============================================================

ordered_targets = [
    targets_per_letter[0]
]


for k in range(
    1,
    n_letters
):

    prev = ordered_targets[-1]

    cur = targets_per_letter[k]

    # Distance between every pair of points
    cost = np.linalg.norm(
        prev[:, None, :] -
        cur[None, :, :],
        axis=2
    )

    row, col = linear_sum_assignment(
        cost
    )

    ordered_targets.append(
        cur[col]
    )


# ===============================================================
# 6. INITIAL RANDOM POSITIONS
# ===============================================================

x0 = rng.uniform(
    -6,
    6,
    size=(N, 2)
)


# Assign initial agents to closest points
cost0 = np.linalg.norm(
    x0[:, None, :] -
    ordered_targets[0][None, :, :],
    axis=2
)

row, col = linear_sum_assignment(
    cost0
)

ordered_targets[0] = (
    ordered_targets[0][col]
)


# Recalculate assignments for the remaining letters
for k in range(
    1,
    n_letters
):

    prev = ordered_targets[k - 1]

    cur = targets_per_letter[k]

    cost = np.linalg.norm(
        prev[:, None, :] -
        cur[None, :, :],
        axis=2
    )

    row, col = linear_sum_assignment(
        cost
    )

    ordered_targets[k] = (
        cur[col]
    )


# ===============================================================
# 7. FORMATION CONTROL PARAMETERS
# ===============================================================

A = nx.to_numpy_array(G)

gamma = 0.08

k_gain = 1.0

dt = 0.03

T_per_letter = 260


# ===============================================================
# 8. SIMULATION
# ===============================================================

x = x0.copy()

history = []

history.append(
    x.copy()
)


for k in range(
    n_letters
):

    xstar = ordered_targets[k]

    print(
        f"Simulating letter {letters[k]}..."
    )

    for t in range(
        T_per_letter
    ):

        # -------------------------------------------------------
        # Relative position of agents
        # -------------------------------------------------------

        diff_x = (
            x[:, None, :]
            -
            x[None, :, :]
        )

        # -------------------------------------------------------
        # Relative target positions
        # -------------------------------------------------------

        diff_xs = (
            xstar[:, None, :]
            -
            xstar[None, :, :]
        )

        # -------------------------------------------------------
        # Formation error
        # -------------------------------------------------------

        err = (
            diff_x -
            diff_xs
        )

        # -------------------------------------------------------
        # Formation control term
        # -------------------------------------------------------

        formation_term = (
            -k_gain *
            np.einsum(
                "ij,ijk->ik",
                A,
                err
            )
        )

        # -------------------------------------------------------
        # Anchor term
        # -------------------------------------------------------

        anchor_term = (
            -gamma *
            (x - xstar)
        )

        # -------------------------------------------------------
        # Total control
        # -------------------------------------------------------

        xdot = (
            formation_term +
            anchor_term
        )

        # -------------------------------------------------------
        # Euler integration
        # -------------------------------------------------------

        x = (
            x +
            dt * xdot
        )

        history.append(
            x.copy()
        )


history = np.array(
    history
)


print(
    "Total simulation frames:",
    history.shape[0]
)


# Save simulation data
np.save(
    "history.npy",
    history
)

np.save(
    "targets.npy",
    np.array(ordered_targets)
)

print("Saved history.npy")
print("Saved targets.npy")


# ===============================================================
# 9. CALCULATE FINAL FORMATION ERROR
# ===============================================================

errs = []

idx = 0


for k in range(
    n_letters
):

    idx += T_per_letter

    xf = history[idx]

    e = np.linalg.norm(
        xf -
        ordered_targets[k],
        axis=1
    ).mean()

    errs.append(e)

    print(
        f"Letter {letters[k]}: "
        f"mean final position error = "
        f"{e:.4f}"
    )


# ===============================================================
# 10. STATIC KUSHAL FIGURE
# ===============================================================

fig, axes = plt.subplots(
    1,
    n_letters,
    figsize=(3 * n_letters, 3.2)
)

idx = 0


for k in range(
    n_letters
):

    idx += T_per_letter

    ax = axes[k]

    xf = history[idx]

    # Plot agents
    ax.scatter(
        xf[:, 0],
        xf[:, 1],
        c="tab:blue",
        s=25
    )

    # Plot communication edges
    for i, j in G.edges():

        ax.plot(
            [xf[i, 0], xf[j, 0]],
            [xf[i, 1], xf[j, 1]],
            color="gray",
            lw=0.4,
            alpha=0.5
        )

    ax.set_title(
        letters[k]
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis("off")


plt.suptitle(
    "Agent formation snapshots: "
    "name KUSHAL, letter by letter"
)

plt.tight_layout()

plt.savefig(
    "kushal_sequence.png",
    dpi=200
)

plt.close()

print(
    "Saved kushal_sequence.png"
)


# ===============================================================
# 11. GENERATE VIDEO
# ===============================================================

print(
    "\nGenerating video..."
)


# Use the same graph
edges = list(
    G.edges()
)


# Subsample frames to keep video size reasonable
step = 3

frames = history[::step]


# Determine which letter is being displayed
letter_of_frame = [
    min(
        i * step // T_per_letter,
        len(letters) - 1
    )
    for i in range(
        len(frames)
    )
]


# Determine plot limits
xmin = (
    history[:, :, 0].min()
    - 1
)

xmax = (
    history[:, :, 0].max()
    + 1
)

ymin = (
    history[:, :, 1].min()
    - 1
)

ymax = (
    history[:, :, 1].max()
    + 1
)


# ---------------------------------------------------------------
# Create animation figure
# ---------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(6, 6)
)


def update(i):

    ax.clear()

    xf = frames[i]

    # -----------------------------------------------------------
    # Draw communication graph
    # -----------------------------------------------------------

    for a, b in edges:

        ax.plot(
            [xf[a, 0], xf[b, 0]],
            [xf[a, 1], xf[b, 1]],
            color="lightgray",
            lw=0.5,
            zorder=1
        )

    # -----------------------------------------------------------
    # Draw agents
    # -----------------------------------------------------------

    ax.scatter(
        xf[:, 0],
        xf[:, 1],
        c="tab:blue",
        s=40,
        zorder=2
    )

    # -----------------------------------------------------------
    # Set plot limits
    # -----------------------------------------------------------

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_ylim(
        ymin,
        ymax
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis("off")

    # -----------------------------------------------------------
    # Display current letter
    # -----------------------------------------------------------

    current_letter = letters[
        letter_of_frame[i]
    ]

    ax.set_title(
        f"Formation control: "
        f"target letter '{current_letter}' "
        f"(name: {NAME})"
    )

    return []


# ===============================================================
# 12. CREATE AND SAVE MP4
# ===============================================================

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=40,
    blit=False
)


# FFmpeg video writer
writer = animation.FFMpegWriter(
    fps=25,
    bitrate=1800
)


# Save video
ani.save(
    "kushal_formation_control.mp4",
    writer=writer,
    dpi=150
)


plt.close()


print(
    "\n======================================"
)

print(
    "Video saved successfully!"
)

print(
    "File: kushal_formation_control.mp4"
)

print(
    "======================================"
)