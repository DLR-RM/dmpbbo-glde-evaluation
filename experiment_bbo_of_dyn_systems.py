import itertools
from pathlib import Path

import numpy as np
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from matplotlib import pyplot as plt

from demo_optimize_dyn_sys_parameters import (TaskFitTrajectory,
                                                                     TaskSolverDmpDynSys,
                                                                     plot_before_after)
from utils import get_demonstration, plot_error_bar


def main_one():
    # 3, 0 very nice for overshooting
    traj = get_demonstration("stulp13learning_meka", traj_number=3)
    d = 0
    traj_demo = Trajectory(traj.ts, traj.ys[:, d], traj.yds[:, d], traj.ydds[:, d])

    # Create task
    goal_cost_weight = 1000.0
    traj_y_final = traj_demo.y_final
    traj_length = traj_demo.length
    task = TaskFitTrajectory(
        traj_y_final, traj_length, goal_cost_weight, plot_trajectories=traj_demo
    )

    # Create task solver
    dmp_types = ["IJSPEERT_2002_MOVEMENT", "KULVICIUS_2012_JOINING", "2022"]
    task_solver = TaskSolverDmpDynSys(traj_demo, dmp_types[2])

    # Run the optimization
    distribution = task_solver.suggest_distribution_init()
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)
    n_updates = 20
    n_samples_per_update = 10
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update
    )

    # Plot results
    plot_before_after(session, traj_demo)
    session.plot()

    plt.show()


def run_optimization_with_traj(traj_demo, dmp_type, decoupled, plot_basename=""):
    # Create task
    goal_cost_weight = 1000.0
    traj_y_final = traj_demo.y_final
    traj_length = traj_demo.length
    task = TaskFitTrajectory(
        traj_y_final, traj_length, goal_cost_weight, plot_trajectories=traj_demo
    )

    # Create task solver
    task_solver = TaskSolverDmpDynSys(traj_demo, dmp_type, decoupled)

    # Run the optimization
    distribution = task_solver.suggest_distribution_init()
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)
    n_updates = 25
    n_samples_per_update = 10
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update
    )
    learning_curve, _ = session.get_learning_curve()

    # Extract the DMPs before and after learning, and train them
    fa_params = {}
    n_updates = session.get_n_updates()
    for label, i_update in {"before": 0, "after": n_updates - 1}.items():
        distribution = session.ask("distribution", i_update)
        task_solver = session.ask("task_solver")
        dmp = task_solver.get_dmp(distribution.mean)

        fas = [FunctionApproximatorRBFN(30, 0.85) for _ in range(dmp.dim_dmp())]
        dmp.train(traj_demo, function_approximators=fas)

        # Save the parameter vector
        dmp.set_selected_param_names("weights")
        fa_params[label] = dmp.get_param_vector()

    if plot_basename:
        session.plot()
        plt.gcf().savefig(f"{plot_basename}_session.png")
        plt.close()

        plot_before_after(session, traj_demo)
        plt.gcf().savefig(f"{plot_basename}_dmps.png")
        plt.close()

    return learning_curve, fa_params


def main_with_demo_dir(dmp_types, demo_type, n_trajs, main_directory, axs=None):
    """Main function for script."""

    trajs = []
    for traj_number in range(0, n_trajs):
        traj = Trajectory.loadtxt(f"data/{demo_type}/traj{traj_number:03}.txt")
        trajs.append(traj)

    decoupleds = [False, True]
    experiments = list(itertools.product(dmp_types, decoupleds))

    if axs is None or len(axs) == 0:
        n_rows = 1
        n_cols = 4
        _, axs = plt.subplots(n_rows, n_cols, figsize=(1.0 * n_cols * 3, n_rows * 3))

    for experiment in experiments:
        dmp_type = experiment[0]
        decoupled = experiment[1]

        dec = "decoupled" if decoupled else "coupled"
        directory = Path(f"{main_directory}/{dmp_type}/{dec}")
        directory.mkdir(parents=True, exist_ok=True)

        costs_all = {"before": [], "after": []}
        fa_params_all = {"before": [], "after": []}

        for i_traj, traj in enumerate(trajs):
            filename = Path(directory, f"{i_traj:03d}_learning_curve.txt")

            if filename.exists():
                print(f"{filename} already exists. Loading.")
                learning_curve = np.loadtxt(filename)
                fa_params = {}
                for ba in ["before", "after"]:
                    filename = Path(directory, f"{i_traj:03d}_fa_params_{ba}.txt")
                    fa_params[ba] = np.loadtxt(filename)

            else:
                print(f"{filename} does not exist. Running optimization.")
                basename = Path(directory, f"{i_traj:03d}")  # For saving plots to
                learning_curve, fa_params = run_optimization_with_traj(
                    traj, dmp_type, decoupled, basename
                )
                np.savetxt(filename, learning_curve)
                for ba in ["before", "after"]:
                    filename = Path(directory, f"{i_traj:03d}_fa_params_{ba}.txt")
                    np.savetxt(filename, fa_params[ba])

            print(f"{filename}: {learning_curve[0][1]} -> {learning_curve[-1][1]}")

            costs_all["before"].append(learning_curve[0][1])
            costs_all["after"].append(learning_curve[-1][1])
            for ba in ["before", "after"]:
                fa_params_all[ba].extend(fa_params[ba])

        i_dmp_type = dmp_types.index(dmp_type)

        ax = axs[0]

        # Results before optimization
        before_mean = np.mean(costs_all["before"])
        plot_error_bar(i_dmp_type, costs_all["before"], "gray", ax)

        # Results after optimization
        after_mean = np.mean(costs_all["after"])
        color = "green" if decoupled else "blue"
        d = 0.6 if decoupled else 0.4
        plot_error_bar(i_dmp_type + d, costs_all["after"], color, ax)

        # Connect before and after at the mean
        ax.plot(
            [i_dmp_type, i_dmp_type + d], [before_mean, after_mean], "-", color=color, linewidth=1
        )

        ylims = {
            "stulp09compact": [0, 19],
            "stulp13learning_meka": [0, 17],
            "coathanger23": [0, 99],
        }
        ylims = {"stulp09compact": [0, 5], "stulp13learning_meka": [0, 5], "coathanger23": [0, 20]}
        # ax.set_ylim(ylims[demo_type])

        ax.set_xticks(range(len(dmp_types)))
        ax.set_xticklabels(dmp_types)
        dmp_type_labels = ["Ijs02", "Kul12", "SCT23"]
        ax.set_xticklabels(dmp_type_labels[: len(dmp_types)])
        ax.set_ylabel("cost")
        ax.set_yscale("log")
        # ax.tick_params(axis="y", direction="in", pad=-25)
        ax.grid(color="#cccccc", linestyle="-", linewidth=0.5)

        # axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        if decoupled:
            ax = axs[1 + i_dmp_type]

            ylims = {
                "stulp09compact": [0, 480],
                "stulp13learning_meka": [0, 790],
                "coathanger23": [0, 640],
            }
            fa_params_lims = {
                "stulp09compact": [-15, 15],
                "stulp13learning_meka": [-15, 15],
                "coathanger23": [-80, 80],
            }

            n_bins = 49

            _, bins, patches = ax.hist(
                fa_params_all["before"],
                bins=n_bins,
                label="before",
                range=fa_params_lims[demo_type],
            )
            for patch in patches:
                patch.set_color("#cccccc")
            counts, bins = np.histogram(
                fa_params_all["after"], bins=bins, range=fa_params_lims[demo_type]
            )
            ax.step(bins[1:], counts, "g")

            print(dmp_type)
            y_scale = 0.8
            for ba in ["before", "after"]:
                mu = np.mean(fa_params_all[ba])
                std = np.std(fa_params_all[ba])
                s = f"    {ba}: {mu:.2f} +- {std:.2f}"
                ax.set_ylim(ylims[demo_type])
                ax.text(0, y_scale * (sum(ax.get_ylim())), s, horizontalalignment="center")
                y_scale -= 0.1
                ax.tick_params(axis="y", direction="in", pad=-25)

                ax.text(
                    0.5 * ax.get_xlim()[1],
                    0.5 * (sum(ax.get_ylim())),
                    dmp_type[:12],
                    horizontalalignment="center",
                )

            ax.set_xlim(fa_params_lims[demo_type])
            ax.axvline(0.0, linestyle="--", color="k")

    plt.gcf().tight_layout()


def main():

    main_directory = "./results/bbo_of_dyn_systems"
    dmp_types = ["IJSPEERT_2002_MOVEMENT", "KULVICIUS_2012_JOINING", "2022"]
    demo_types = ["stulp09compact", "stulp13learning_meka", "coathanger23"]

    n_trajs = 12
    n_rows = 1 + len(dmp_types)
    n_cols = len(demo_types)
    _, axs = plt.subplots(n_rows, n_cols, figsize=(1.0 * n_cols * 3, n_rows * 3))

    for i_demo, demo_type in enumerate(demo_types):
        cur_axs = [axs[i][i_demo] for i in range(n_rows)]
        directory = Path(main_directory, demo_type)
        main_with_demo_dir(dmp_types, demo_type, n_trajs, directory, cur_axs)

    filename = Path(main_directory, f"results.svg")
    print(f"Saving to {filename}")
    plt.gcf().savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
