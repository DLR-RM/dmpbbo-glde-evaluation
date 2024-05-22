import numpy as np
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.dmps.Trajectory import Trajectory
from matplotlib import pyplot as plt

from demo_optimize_dyn_sys_parameters import (TaskFitTrajectory,
                                                                     TaskSolverDmpDynSys)
from save_plot import save_plot
from utils import get_demonstration


def main_optimization(traj_demo, dmp_type="2022", n_updates=30, n_samples_per_update=10):
    # Create task
    goal_cost_weight = 100.0
    traj_y_final = traj_demo.y_final
    traj_length = traj_demo.length
    task = TaskFitTrajectory(
        traj_y_final, traj_length, goal_cost_weight, plot_trajectories=traj_demo
    )

    # Create task solver
    task_solver = TaskSolverDmpDynSys(traj_demo, dmp_type)

    # Run the optimization
    distribution = task_solver.suggest_distribution_init()
    if dmp_type == "IJSPEERT_2002_MOVEMENT":
        distribution.mean[0] = 50
    if dmp_type == "SCT23":
        distribution.mean[2] = 25  # Otherwise fitting is quite good initially.
        distribution.mean[3] = 0.03  # Otherwise fitting is quite good initially.
    print(distribution.mean)

    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.85)
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update
    )

    # session.plot()
    n_updates = session.get_n_updates()
    for i_update in range(n_updates):
        n_plots = 1
        fig, axs = plt.subplots(1, n_plots)
        if not isinstance(axs, list):
            axs = [axs]

        session.plot_rollouts_update(i_update, plot_eval=True, plot_samples=True, ax=axs[0])
        axs[0].set_ylim([-0.3, 4])

        if n_plots > 1 and i_update > 0:
            session.plot_learning_curve(ax=axs[1], n_updates=i_update + 1)
            axs[1].set_xlim([0, n_updates * n_samples_per_update])
            axs[1].set_ylim([0, 13])

        save_plot(f"presentation_optimization_{dmp_type}_{i_update:02}.png", directory="plots")


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

    dmp_types = ["IJSPEERT_2002_MOVEMENT", "KULVICIUS_2012_JOINING", "SCT23"]
    for dmp_type in dmp_types:
        main_optimization(traj_demo, dmp_type, n_updates=30, n_samples_per_update=10)
    # plt.show()


if __name__ == "__main__":
    main()
