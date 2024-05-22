""" Module to illustrate the Richard's system, aka the generalized logistic differential equation. """
import numpy as np
from dmpbbo.dynamicalsystems.RichardsSystem import RichardsSystem
from matplotlib import pyplot as plt

from save_plot import save_plot


def plot_dyn_sys(dyn_sys, axs=None, plot_past=True):
    """
    Plot a dynamical system.

    @param dyn_sys: The dynamical system
    @param axs: The axes to plot on
    @param plot_past: Whether to plot before t=0 also.
    @return: The axes and the color used for plotting.
    """
    tau = dyn_sys.tau

    xlims = [0.0, tau * 1.3]

    # Graph after t=0
    ts = np.linspace(0.0, xlims[1], 151)
    xs, xds = dyn_sys.integrate(ts)
    lines, axs = dyn_sys.plot(ts, xs, xds, axs=axs)

    lines_past = None
    if plot_past:
        # Graph before t=0
        xlims[0] = -0.5 * tau
        ts = np.linspace(xlims[0], xlims[1], 201)
        xs, xds = dyn_sys.analytical_solution(ts)
        lines_past, _ = dyn_sys.plot(ts, xs, xds, axs=axs)
        lines.extend(lines_past)

    # Give all lines the same style
    color = lines[0].get_color()
    plt.setp(lines, color=color, alpha=1.0, linewidth=1.5)
    if plot_past:
        plt.setp(lines_past, alpha=0.5)

    for ax in axs:
        ax.axvline(0.0, color="k")
        ax.axvline(tau, color="k")
        ax.set_xlim(xlims)

    return axs, color


def main():
    """
    Main function.
    """

    tau = 6.0
    x_left = 0.0
    x_init = 0.2
    x_attr = 5.0
    v = 2.0
    alpha = 10.0

    axs = None
    for alpha_new in np.linspace(4, 8, 5):
        dyn_sys = RichardsSystem(tau, x_init, x_attr, 1.0, alpha=alpha_new, v=v)
        dyn_sys.set_left_asymp(x_left)
        axs, _ = plot_dyn_sys(dyn_sys, axs=axs, plot_past=True)
    axs[0].set_ylim([x_left - 0.1, x_attr + 0.1])
    # save_plot("richards_system_alpha.svg")

    x_init = 3.0
    x_attr = 5.0

    axs = None
    for d_x_left in [-0.5, -0.1, -0.02, -0.004]:
        dyn_sys = RichardsSystem(tau, x_init, x_attr, 0.0, alpha=alpha, v=v)
        dyn_sys.set_left_asymp(x_init + d_x_left)
        axs, _ = plot_dyn_sys(dyn_sys, axs=axs, plot_past=True)
    axs[0].set_ylim([x_init - 0.5 - 0.1, x_attr + 0.1])
    # save_plot("richards_system_left_asymptote.svg")

    axs = None
    for t_infl in range(1, 6):
        t_infl_ratio = t_infl / tau
        dyn_sys = RichardsSystem(tau, x_init, x_attr, t_infl_ratio, alpha=alpha, v=v)
        axs, color = plot_dyn_sys(dyn_sys, axs=axs, plot_past=True)

        # Plot vertical line at inflection time
        x, xd = dyn_sys.analytical_solution(np.array([t_infl]))
        axs[0].plot([t_infl, t_infl], [0, x], "--", color=color)
        axs[0].plot(t_infl, x, "o", color=color)
        axs[1].plot([t_infl, t_infl], [0, xd], "--", color=color)
        axs[1].plot(t_infl, xd, "o", color=color)

    axs[0].set_ylim([x_init - 0.5 - 0.1, x_attr + 0.1])
    axs[0].set_ylim([2.6, 5.1])
    axs[1].set_ylim([0.0, 1.6])
    save_plot("richards_system_t_infl.svg")

    x_init = 1.0
    x_attr = 0.0
    v = 10.0
    axs = None
    for t_infl in range(1, 6):
        t_infl_ratio = t_infl / tau
        dyn_sys = RichardsSystem(tau, x_init, x_attr, t_infl_ratio, alpha=alpha, v=v)
        axs, color = plot_dyn_sys(dyn_sys, axs=axs, plot_past=True)
        for ax in axs:
            ax.axvline(t_infl, color=color)
    axs[0].set_ylim([-0.1, 1.1])
    # save_plot("richards_system_t_infl_decreasing.svg")

    plt.show()


if __name__ == "__main__":
    main()
