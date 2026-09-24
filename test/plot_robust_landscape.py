"""绘制鲁棒优化校准函数的目标面、约束边界和候选点。

运行：python test/plot_robust_landscape.py
输出：test/robust_landscape.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).with_name("robust_landscape.png")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


def f_obj(x1, x2):
    background = 20.0 + 0.03 * ((x1 - 2.0) ** 2 + (x2 - 2.0) ** 2)
    narrow = -8.0 * np.exp(-0.5 * ((x1 - 3.9) ** 2 + (x2 - 3.9) ** 2) / 0.18 ** 2)
    broad = -11.5 * np.exp(-0.5 * ((x1 - 2.0) ** 2 + (x2 - 2.0) ** 2) / 2.0 ** 2)
    return background + narrow + broad


def g_constr(x1, x2):
    return x1 + x2 - 8.0


def main():
    x = np.linspace(0.0, 4.0, 320)
    x1, x2 = np.meshgrid(x, x)
    z = f_obj(x1, x2)
    constraint = g_constr(x1, x2)

    nominal = np.array([3.9, 3.9])
    stable = np.array([2.0, 2.0])
    sigma = 4.0 / (2.0 * 6.0)
    theta = np.linspace(0.0, 2.0 * math.pi, 160)
    uncertainty_x = nominal[0] + sigma * np.cos(theta)
    uncertainty_y = nominal[1] + sigma * np.sin(theta)

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    formula = (
        r"$f(x)=20+0.03\|(x_1,x_2)-(2,2)\|^2$" "\n"
        r"$\quad-8\,\mathrm{exp}[-\|x-(3.9,3.9)\|^2/(2\cdot0.18^2)]$" "\n"
        r"$\quad-11.5\,\mathrm{exp}[-\|x-(2,2)\|^2/(2\cdot2.0^2)]$"
    )
    point_note = (
        "* narrow nominal point: (3.9, 3.9), margin ≈ 0.2\n"
        "P robust point: (2.0, 2.0), margin ≈ 4.0"
    )

    contour = axes[0].contourf(x1, x2, z, levels=40, cmap="viridis")
    axes[0].contour(x1, x2, z, levels=14, colors="white", linewidths=0.35, alpha=0.55)
    axes[0].contour(x1, x2, constraint, levels=[0], colors="#ef4444", linewidths=2.5)
    axes[0].plot(*nominal, "*", ms=15, color="#dc2626")
    axes[0].plot(*stable, "P", ms=12, color="#16a34a")
    axes[0].plot(uncertainty_x, uncertainty_y, "--", color="#f97316", lw=1.5,
                  label="Nominal-point input perturbation (±1σ)")
    axes[0].set_title("Objective landscape and constraint boundary")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].text(0.03, 0.97, formula, transform=axes[0].transAxes,
                 va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.86})
    axes[0].text(0.03, 0.03, point_note, transform=axes[0].transAxes,
                 va="bottom", fontsize=8,
                 bbox={"facecolor": "white", "alpha": 0.86})
    fig.colorbar(contour, ax=axes[0], label="$f(x_1,x_2)$ (lower is better)")

    axes[1].contourf(x1, x2, constraint, levels=30, cmap="coolwarm", alpha=0.9)
    axes[1].contour(x1, x2, constraint, levels=[0], colors="black", linewidths=2.5)
    axes[1].fill_between(x, 0, 8.0 - x, color="#22c55e", alpha=0.18,
                         label="Feasible region $g(x)\\leq0$")
    axes[1].plot(*nominal, "*", ms=15, color="#dc2626")
    axes[1].plot(*stable, "P", ms=12, color="#16a34a")
    axes[1].set_xlim(0, 4)
    axes[1].set_ylim(0, 4)
    axes[1].set_title("Constraint function and safety margin")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].text(0.04, 0.96, r"$g(x_1,x_2)=x_1+x_2-8.0\leq0$",
                 transform=axes[1].transAxes, va="top", fontsize=11,
                 bbox={"facecolor": "white", "alpha": 0.86})
    axes[1].text(0.04, 0.04, "Green area: feasible region\nRed line: g(x)=0",
                 transform=axes[1].transAxes, va="bottom", fontsize=8,
                 bbox={"facecolor": "white", "alpha": 0.86})

    surface = axes[2].plot_surface(x1, x2, z, cmap="viridis", linewidth=0,
                                   antialiased=True, alpha=0.92)
    axes[2].plot([nominal[0]], [nominal[1]], [f_obj(*nominal)], "*",
                 color="#dc2626", ms=14)
    axes[2].plot([stable[0]], [stable[1]], [f_obj(*stable)], "P",
                 color="#16a34a", ms=10)
    axes[2].set_title("3D objective surface")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")
    axes[2].set_zlabel("$f(x_1,x_2)$")
    axes[2].view_init(elev=32, azim=-125)
    axes[2].text2D(0.02, 0.96, formula, transform=axes[2].transAxes,
                   va="top", fontsize=8,
                   bbox={"facecolor": "white", "alpha": 0.86})

    constraint_surface = axes[3].plot_surface(
        x1, x2, constraint, cmap="coolwarm", linewidth=0, alpha=0.88
    )
    axes[3].plot_surface(x1, x2, np.zeros_like(constraint), color="black", alpha=0.18)
    axes[3].plot([nominal[0]], [nominal[1]], [g_constr(*nominal)], "*",
                 color="#dc2626", ms=14)
    axes[3].plot([stable[0]], [stable[1]], [g_constr(*stable)], "P",
                 color="#16a34a", ms=10)
    axes[3].set_title("3D constraint surface")
    axes[3].set_xlabel("$x_1$")
    axes[3].set_ylabel("$x_2$")
    axes[3].set_zlabel("$g(x_1,x_2)$")
    axes[3].view_init(elev=28, azim=-125)
    axes[3].text2D(0.03, 0.94, "$g(x_1,x_2)=x_1+x_2-8.0\\leq0$\nblack plane: $g=0$",
                   transform=axes[3].transAxes, va="top", fontsize=10,
                   bbox={"facecolor": "white", "alpha": 0.86})

    fig.suptitle(
        "Robust optimization calibration: nominally better but fragile vs. stable near-optimum",
        fontsize=14,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"已生成：{OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
