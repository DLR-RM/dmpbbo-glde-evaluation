import numpy as np
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.dynamicalsystems import SigmoidSystem
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.RichardsSystem import RichardsSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dmpbbo.dynamicalsystems.TimeSystem import TimeSystem
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from matplotlib import pyplot as plt

from dmpbbo_sct_experiments.save_plot import save_plot
from dmpbbo_sct_experiments.utils import get_demonstration


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


def main_training(traj_demo, n_samples=20):
    tau = traj_demo.duration
    y_init = traj_demo.y_init
    y_attr = traj_demo.y_final
    n_time_steps = 101
    ts = np.linspace(0.0, 1.15 * tau, n_time_steps)

    plot_jerk = False
    names = ["const", "exponential", "sigmoid"]

    n_cols = 5
    if plot_jerk:
        n_cols += 1
    n_rows = len(names)

    # fig, all_axs = plt.subplots(n_rows, n_cols, figsize=(1.3 * 2.5 * n_cols, 2.5 * n_rows))
    fig, all_axs = plt.subplots(n_rows, n_cols, figsize=(0.8 * 16, 9))
    for row, name in enumerate(names):

        axs = all_axs[row]
        markers_max = []

        for i_sample in range(n_samples):
            mean_sample = i_sample > n_samples - 2
            mimic_kulvicius = False  # i_sample == 0

            dmp_args = {}

            print(name)
            if name == "const":
                damp_coef = 10.0 if mean_sample else np.random.uniform(5.0, 20.0)
                dmp_args["transformation_system"] = SpringDamperSystem(
                    tau, y_init, y_attr, damp_coef
                )
                dmp_args["goal_system"] = None

                dmp_args["gating_system"] = ExponentialSystem(tau, 1, 0, 4)
                dmp_args["phase_system"] = ExponentialSystem(tau, 1, 0, 4)

            elif name in ["exponential"]:
                damp_coef = 15.0 if mean_sample else np.random.uniform(7.5, 30.0)
                alpha = 5.0 if mean_sample else np.random.uniform(2.5, 10.0)
                dmp_args["transformation_system"] = SpringDamperSystem(
                    tau, y_init, y_attr, damp_coef
                )
                dmp_args["goal_system"] = ExponentialSystem(tau, y_init, y_attr, alpha)
                y_tau_0_ratio = 0.1
                dmp_args["gating_system"] = SigmoidSystem.SigmoidSystem.for_gating(
                    tau, y_tau_0_ratio
                )
                dmp_args["phase_system"] = TimeSystem(tau, count_down=True)
            else:
                # Values below found through optimization.
                damp_coef = 23.5 if mean_sample else np.random.uniform(11.8, 47.0)
                t_infl_ratio = 0.30 if mean_sample else np.random.uniform(0.15, 0.6)
                alpha = 12.2 if mean_sample else np.random.uniform(6.1, 24.4)
                v = 1.18 if mean_sample else np.random.uniform(1.0, 2.36)
                if i_sample == 0:
                    damp_coef = 20
                    alpha = 20
                    t_infl_ratio = 0.45
                    v = 1

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

                dmp_args["gating_system"] = RichardsSystem(
                    tau, np.ones((1,)), np.zeros((1,)), 1.0, 10.0, 10.0
                )
                dmp_args["phase_system"] = TimeSystem(tau, count_down=True)

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

            # Samples have been plot. Now plot demo trajectory.
            h, _ = traj_demo.plot(axs[1:4])
            if plot_jerk:
                h_jerk = axs[4].plot(traj_demo.ts, traj_demo.yddds())
                h.append(h_jerk)
            plt.setp(h, color="#cad55c", linestyle="-", linewidth=6.0, alpha=0.8, zorder=1)

            use_fas = [False, True] if mean_sample and n_samples < 3 else [False]
            for use_fa in use_fas:
                if use_fa:
                    dmp_args["save_training_data"] = True
                    function_apps = [FunctionApproximatorRBFN(10, 0.7)]
                    dmp = Dmp.from_traj(traj_demo, function_apps, **dmp_args)
                else:
                    dmp = Dmp(tau, y_init, y_attr, None, **dmp_args)

                xs, xds, forcing_terms, fa_outputs = dmp.analytical_solution(ts)
                traj = dmp.states_as_trajectory(ts, xs, xds)
                handles = []  # noqa
                if not use_fa:
                    handles.append(axs[0].plot(ts, xs[:, dmp.GOAL]))
                handles.append(axs[1].plot(ts, traj.ys))
                handles.append(axs[2].plot(ts, traj.yds))
                handles.append(axs[3].plot(ts, traj.ydds))
                if plot_jerk:
                    handles.append(axs[4].plot(ts, traj.yddds()))
                if use_fa:
                    handles.append(axs[-1].plot(ts, fa_outputs))
                    # hs, _ = dmp._function_approximators[0].plot(ax=axs[-1], plot_residuals=False, plot_model_parameters=True)
                    # plt.setp(hs, color="g", linewidth=2)
                    # axs[-1].invert_xaxis()

                if n_samples < 3 and i_sample > n_samples - 3:
                    if use_fa:
                        plt.setp(handles, color="#cb6544", linewidth=2.0)
                    else:
                        plt.setp(handles, color="#3b98cb", linewidth=2.0)
                else:
                    plt.setp(handles, color="#3b98cb", linewidth=0.5)

        labels = [r"$\hat{g}~(m)$", r"$y~(m)$", r"$\dot{y}~(m/s)$", r"$\ddot{y}~(m/s^2)$"]
        if plot_jerk:
            labels.append(r"$\dddot{y}~(m/s^3)$")
        labels.append(r"$f_\mathbf{\theta}~(m/s^2)$")
        for ax, label in zip(axs, labels):
            ax.set_xlabel("time (s)")
            # ax.set_ylabel("")
            ax.set_ylabel(label)
            ax.set_xlim([ts[0], ts[-1]])
            # ax.autoscale(enable=True, axis='x', tight=True)

        for ax in axs:
            # ax.axvline(tau, color="#999999")
            ax.set_facecolor("#fafafa")
            ax.tick_params(color="#bbbbbb", labelcolor="#bbbbbb")
            for spine in ax.spines.values():
                spine.set_edgecolor("#bbbbbb")

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

    if plot_jerk:
        i_col = 4
        # Set max ranges on all axes in the column
        # for i_row in range(n_rows):
        #    all_axs[i_row][i_col].set_ylim([-23000, 14000])

    # Add 0 line
    for i_row in range(n_rows):
        for i_col in range(2, n_cols):
            all_axs[i_row][i_col].axhline(0, color="#bbbbbb")

    save_plot(f"presentation_training_{n_samples:02}.svg", directory="plots")


def main():
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=300)

    demo_name = "stulp09compact"
    # demo_name = "stulp13learning_meka"
    traj_number = 7
    traj_demo = get_demonstration(demo_name, traj_number=traj_number)
    i_dim = 1
    traj_demo = Trajectory(
        traj_demo.ts, traj_demo.ys[:, i_dim], traj_demo.yds[:, i_dim], traj_demo.ydds[:, i_dim]
    )

    main_training(traj_demo, 2)
    main_training(traj_demo, 20)
    plt.show()


if __name__ == "__main__":
    main()
