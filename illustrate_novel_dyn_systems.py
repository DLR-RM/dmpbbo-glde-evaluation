import numpy as np
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.RichardsSystem import RichardsSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from matplotlib import pyplot as plt

from save_plot import save_plot
from utils import get_demonstration, get_function_apps


def plot_dyn_sys(axs, **kwargs):
    damping_coefficient = kwargs.get("damping_coefficient", 20.0)
    spring_constant = kwargs.get("spring_constant", 100)
    mass = kwargs.get("mass", 1.0)

    tau = 1.0
    y_init = np.array([0.0])
    y_attr = np.array([1.0])
    ds = SpringDamperSystem(tau, y_init, y_attr, damping_coefficient, spring_constant, mass)

    ts = np.linspace(0.0, 1.5 * tau, 100)
    xs, xds = ds.integrate(ts)
    line_handles, _ = ds.plot(ts, xs, xds, axs=axs)
    return line_handles


def plot_dyn_sys_default(axs):
    line_handles = plot_dyn_sys(axs)
    plt.setp(line_handles, color="k", linewidth=3, linestyle="--")


def main_spring_damper_parameters():
    fig = plt.figure()

    axs = [fig.add_subplot(330 + i) for i in [1, 2, 3]]
    values = np.linspace(0.5, 10.5, 11)
    for mass in values:
        plot_dyn_sys(axs, **{"mass": mass})
    plot_dyn_sys_default(axs)
    axs[0].legend([f"{v:.2f}" for v in values])
    axs[0].set_title("mass")

    axs = [fig.add_subplot(330 + i) for i in [4, 5, 6]]
    values = np.linspace(2, 200, 10)
    for spring_constant in values:
        plot_dyn_sys(axs, **{"spring_constant": spring_constant})
    plot_dyn_sys_default(axs)
    axs[0].legend([f"{v:.0f}" for v in values])
    axs[0].set_title("spring constant")

    axs = [fig.add_subplot(330 + i) for i in [7, 8, 9]]
    values = np.linspace(0, 30, 10)
    for damping_coefficient in values:
        plot_dyn_sys(axs, **{"damping_coefficient": damping_coefficient})
    plot_dyn_sys_default(axs)
    axs[0].legend([f"{v:.0f}" for v in values])
    axs[0].set_title("damping coefficient")


def plot_dmp(tau, y_init, y_attr, transf_system, goal_system, ts, axs):
    dmp_args = {"transformation_system": transf_system, "goal_system": goal_system}
    dmp = Dmp(tau, y_init, y_attr, None, **dmp_args)
    dmp.plot(ts, axs=axs, plot_tau=False)

    xs, xds, _, _ = dmp.analytical_solution(ts)

    # Compute and plot jerk also
    ydds = xds[:, 1 * dmp.dim_y : 2 * dmp.dim_y] / dmp.tau  # acc
    yddds = np.gradient(ydds.squeeze(), ts)  # jerk
    axs[4].plot(ts, yddds)
    axs[4].set_ylabel(r"$jerk y$")
    axs[4].set_xlabel(r"time ($s$)")
    axs[4].grid()

    for ax in axs:
        ax.axis("tight")


def main_sigmoid_goal_system():
    demo_name = "stulp09compact"
    # demo_name = "stulp13learning_meka"
    traj_number = 7
    traj_demo = get_demonstration(demo_name, traj_number=traj_number)
    i_dim = 1
    traj_demo = Trajectory(
        traj_demo.ts, traj_demo.ys[:, i_dim], traj_demo.yds[:, i_dim], traj_demo.ydds[:, i_dim]
    )

    tau = traj_demo.duration
    y_init = traj_demo.y_init
    y_attr = traj_demo.y_final
    ts = np.linspace(0.0, 1.15 * tau, 101)

    n_samples = 20

    plot_damping = False
    names = ["const", "exponential", "sigmoid"]
    if plot_damping:
        names.append("sigmoid_damping")

    n_cols = 6 if plot_damping else 5
    n_rows = len(names)

    fig, all_axs = plt.subplots(n_rows, n_cols, num=8, figsize=(1.3 * 2.5 * n_cols, 2.5 * n_rows))
    for row, name in enumerate(names):

        axs = all_axs[row]
        markers_max = []

        for i_sample in range(n_samples):
            mean_sample = i_sample == n_samples - 1
            mimic_kulvicius = i_sample == 0

            dmp_args = {}

            print(name)
            if name == "const":
                damp_coef = 10.0 if mean_sample else np.random.uniform(5.0, 20.0)
                dmp_args["transformation_system"] = SpringDamperSystem(
                    tau, y_init, y_attr, damp_coef
                )
                dmp_args["goal_system"] = None
            elif name in ["exponential"]:
                damp_coef = 15.0 if mean_sample else np.random.uniform(7.5, 30.0)
                alpha = 5.0 if mean_sample else np.random.uniform(2.5, 10.0)
                dmp_args["transformation_system"] = SpringDamperSystem(
                    tau, y_init, y_attr, damp_coef
                )
                dmp_args["goal_system"] = ExponentialSystem(tau, y_init, y_attr, alpha)
            else:
                # Values below found through optimization.
                damp_coef = 23.5 if mean_sample else np.random.uniform(11.8, 47.0)
                t_infl_ratio = 0.30 if mean_sample else np.random.uniform(0.15, 0.6)
                alpha = 12.2 if mean_sample else np.random.uniform(6.1, 24.4)
                v = 1.18 if mean_sample else np.random.uniform(1.0, 2.36)
                if mimic_kulvicius:
                    t_infl_ratio = 0.0
                    alpha = 5.0
                    v = 1.0
                    damp_coef = 15.0

                dmp_args["transformation_system"] = SpringDamperSystem(
                    tau, y_init, y_attr, damp_coef
                )
                dmp_args["goal_system"] = RichardsSystem(
                    tau, y_init, y_attr, t_infl_ratio, alpha, v
                )
                dmp_args["dmp_type"] = "2022_NO_SCALING"
                if "damping" in name:
                    print(dmp_args["transformation_system"].damping_coefficient)
                    print(dmp_args["transformation_system"].spring_constant)
                    sc = (20.0 * 20.0) / 4.0 if mean_sample else np.random.uniform(50, 200.0)
                    alpha = 10.0 if mean_sample else np.random.uniform(5.0, 20.0)
                    dmp_args["transformation_system"].spring_constant = sc
                    damping_final = dmp_args["transformation_system"].damping_coefficient
                    damping_init = 0.1 * damping_final
                    dmp_args["damping_system"] = ExponentialSystem(
                        tau, damping_init, damping_final, alpha
                    )

            dmp = Dmp(tau, y_init, y_attr, None, **dmp_args)

            xs, xds, _, _ = dmp.analytical_solution(ts)
            traj = dmp.states_as_trajectory(ts, xs, xds)
            handles = []  # noqa
            handles.append(axs[0].plot(ts, xs[:, dmp.GOAL]))
            handles.append(axs[1].plot(ts, traj.ys))
            handles.append(axs[2].plot(ts, traj.yds))
            handles.append(axs[3].plot(ts, traj.ydds))
            handles.append(axs[4].plot(ts, traj.yddds()))
            if plot_damping:
                handles.append(axs[5].plot(ts, xs[:, dmp.DAMPING]))

            if plot_damping:
                i = np.argmax(traj.ys)
                markers_max.extend(axs[1].plot(ts[i], traj.ys[i], "+g"))
            i = np.argmax(np.abs(traj.yds))
            markers_max.extend(axs[2].plot(ts[i], traj.yds[i], "+g"))
            i = np.argmax(np.abs(traj.ydds))
            markers_max.extend(axs[3].plot(ts[i], traj.ydds[i], "+g"))
            i = np.argmax(np.abs(traj.yddds()))
            markers_max.extend(axs[4].plot(ts[i], traj.yddds()[i], "+g"))
            plt.setp(markers_max, color="g")

            if mean_sample:
                plt.setp(handles, color="b", linewidth=2.0)
            else:
                plt.setp(handles, color="#777777", linewidth=0.5)

        # Samples have been plot. Now plot demo trajectory.
        h, _ = traj_demo.plot(axs[1:4])
        h_jerk = axs[4].plot(traj_demo.ts, traj_demo.yddds())
        h.append(h_jerk)
        plt.setp(h, color="r")

        labels = [
            r"$\hat{g}~(m)$",
            r"$y~(m)$",
            r"$\dot{y}~(m/s)$",
            r"$\ddot{y}~(m/s^2)$",
            r"$\dddot{y}~(m/s^3)$",
        ]
        if plot_damping:
            labels.append("damp")
        for ax, label in zip(axs, labels):
            ax.set_xlabel("time (s)")
            ax.set_ylabel(label)
            ax.set_xlim([ts[0], ts[-1]])
            # ax.autoscale(enable=True, axis='x', tight=True)

        for ax in axs:
            ax.axvline(tau, color="#bbbbbb")
            ax.set_facecolor("#f7f7f7")

        for ax in axs:
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position("both")
            for label in ax.yaxis.get_ticklabels():
                label.set_horizontalalignment("right")
            ax.tick_params(axis="y", direction="in", pad=-5)
        if row < n_rows - 1:
            for ax in axs:
                ax.set_xlabel("")
                ax.set_xticks([])

    # Share ylims between plots to facilitate visual comparison
    for i_col in range(1, n_cols):
        ylims = (np.infty, -np.infty)
        # Get max ranges
        for i_row in range(n_rows):
            cur_ylims = all_axs[i_row][i_col].get_ylim()
            ylims = (min(ylims[0], cur_ylims[0]), max(ylims[1], cur_ylims[1]))
        # Set max ranges on all axes in the column
        for i_row in range(n_rows):
            all_axs[i_row][i_col].set_ylim(ylims)
            if i_col == 1:  # i_col==1 and i_col==0 share the same axes
                all_axs[i_row][0].set_ylim(ylims)

    i_col = 4
    # Set max ranges on all axes in the column
    for i_row in range(n_rows):
        all_axs[i_row][i_col].set_ylim([-23000, 14000])
        pass


def main_towards_critical_damping_illustrate():
    n_dims = 1
    tau = 0.8
    y_init = np.array([2.0])
    y_attr = np.array([0.0])

    # Prepare memory
    ts = np.linspace(0.0, 1.5 * tau, 100)
    n_time_steps = len(ts)
    xs_g = np.empty((n_time_steps, n_dims))
    xds_g = np.empty((n_time_steps, n_dims))
    xs_s = np.empty((n_time_steps, 2 * n_dims))
    xds_s = np.empty((n_time_steps, 2 * n_dims))

    spring_system = SpringDamperSystem(tau, y_init, y_attr, 20)
    damp_coef_crit = spring_system.damping_coefficient
    damp_coef_low = 0.3 * spring_system.damping_coefficient

    alpha = 3
    damp_systems = {
        "critical": ExponentialSystem(tau, 0.99999 * damp_coef_crit, damp_coef_crit, alpha),
        "low": ExponentialSystem(tau, 0.99999 * damp_coef_low, damp_coef_low, alpha),
        "decaying": ExponentialSystem(tau, damp_coef_low, damp_coef_crit, alpha),
    }

    n_rows = 1
    n_cols = 4
    ratio = 1.61
    fig = plt.figure(figsize=(ratio * n_cols * 3, n_rows * 3))
    axs = [fig.add_subplot(n_rows, n_cols, ii + 1) for ii in range(n_rows * n_cols)]

    for name, damp_system in damp_systems.items():

        xs_g[0, :], xds_g[0, :] = damp_system.integrate_start()
        spring_system.damping_coefficient = xs_g[0, :]
        xs_s[0, :], xds_s[0, :] = spring_system.integrate_start()
        for ii in range(1, n_time_steps):
            dt = ts[ii] - ts[ii - 1]
            xs_g[ii, :], xds_g[ii, :] = damp_system.integrate_step_runge_kutta(dt, xs_g[ii - 1, :])
            spring_system.damping_coefficient = xs_g[ii, :]
            xs_s[ii, :], xds_s[ii, :] = spring_system.integrate_step_runge_kutta(
                dt, xs_s[ii - 1, :]
            )

        axs[0].plot(ts, xs_g)
        spring_system.plot(ts, xs_s, xds_s, axs=axs[1:4])

    axs[0].legend(damp_systems.keys())
    for i, label in enumerate(["g", "y", "yd", "ydd"]):
        ax = axs[i]
        ax.set_xlim([ts[0], ts[-1]])
        ax.set_xlabel("time (s)")
        ax.set_ylabel(label)

    plt.gcf().canvas.set_window_title("towards_critical_damping_illustration")


def main_towards_critical_damping_train():
    # Show on real trajectory
    demo_name = "stulp13learning_meka"
    traj_number = 4
    traj_demo = get_demonstration(demo_name, traj_number=traj_number)
    n_dofs = traj_demo.dim

    for dmp_type in ["KULVICIUS_2012_JOINING", "2022"]:
        function_apps = get_function_apps(n_dofs, "RBFN", 30)
        dmp = Dmp.from_traj(traj_demo, function_apps, dmp_type=dmp_type)
        dmp.plot(plot_no_forcing_term_also=True, plot_demonstration=traj_demo)
        plt.gcf().canvas.set_window_title(dmp_type)


def main():
    # main_spring_damper_parameters()
    # save_plot("illustrate_spring_damper_parameters.svg")

    main_sigmoid_goal_system()
    save_plot("illustrate_sigmoid_goal_system.svg")

    # main_towards_critical_damping_illustrate()
    # save_plot("illustrate_towards_critical_damping.svg")

    # main_towards_critical_damping_train()
    # save_plot("illustrate_towards_critical_damping_training.svg")

    plt.show()


if __name__ == "__main__":
    main()
