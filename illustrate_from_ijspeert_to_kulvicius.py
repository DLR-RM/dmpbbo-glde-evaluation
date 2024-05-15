""" Module to plot the effect of going from the Ijspeert to the Kulvicius formulation. """
import tikzplotlib

from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.RichardsSystem import RichardsSystem
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem
from dmpbbo.dynamicalsystems.TimeSystem import TimeSystem

from dmpbbo_sct_experiments.utils import *


def main():
    """Run one demo for bbo_of_dmps.
    """

    demo_name = "stulp09compact"
    # demo_name = "stulp13learning_meka"
    traj_number = 2
    traj_demo = get_demonstration(demo_name, traj_number=traj_number)
    i_dim = 1
    traj_demo = Trajectory(
        traj_demo.ts, traj_demo.ys[:, i_dim], traj_demo.yds[:, i_dim], traj_demo.ydds[:, i_dim]
    )

    n_basis = 10
    n_dims = traj_demo.dim
    tau = traj_demo.duration
    y_init = traj_demo.y_init
    y_attr = traj_demo.y_final

    all_dmp_args = [
        {
            "dmp_type": "IJSPEERT_2002_MOVEMENT",
            "alpha_spring_damper": 10,
            "phase_system": TimeSystem(traj_demo.duration, True),
        },
        {
            "dmp_type": "KULVICIUS_2012_JOINING",
            "alpha_spring_damper": 15,
            "goal_system": ExponentialSystem(tau, y_init, y_attr, 5),
            "gating_system": SigmoidSystem(tau, 1, -100.0, 1.0),
        },
        {
            "dmp_type": "2022_NO_SCALING",
            "alpha_spring_damper": 15,
            "goal_system": RichardsSystem(tau, y_init, y_attr, 0.26, 12.0, 2.0),
            "gating_system": RichardsSystem(tau, np.ones((1,)), np.zeros((1,)), 1.0, 10.0, 10.0),
        },
    ]

    ts = np.arange(0.0, 1.25 * traj_demo.duration, traj_demo.dt_mean)

    n_cols = 5
    n_rows = len(all_dmp_args)
    fig = plt.figure(figsize=(2 * n_cols, 3 * n_rows))
    all_axs = [fig.add_subplot(n_rows, n_cols, i + 1) for i in range(n_rows * n_cols)]

    for i_row, dmp_args in enumerate(all_dmp_args):
        axs = all_axs[0 + i_row * n_cols : 5 + i_row * n_cols]

        function_apps = [FunctionApproximatorRBFN(n_basis, 0.9) for _ in range(n_dims)]
        dmp = Dmp.from_traj(traj_demo, function_apps, **dmp_args)

        h, axs = dmp.plot(
            ts,
            axs=axs,
            plot_demonstration=traj_demo,
            plot_no_forcing_term_also=True,
            plot_compact=True,
        )
        plt.setp(h, linewidth=4)
        axs[0].set_ylabel("\ensuremath{\\vy}")
        axs[1].set_ylabel("\ensuremath{\dot{\\vy} = \\vz/\\tau}")
        axs[1].set_ylim([-0.2, 2.4])
        axs[2].set_ylabel("\ensuremath{\ddot{\\vy} = \ddot{\\vz}/\\tau}")
        axs[2].set_ylim([-24, 35])
        axs[3].set_ylabel("\ensuremath{\\canon\\fa(\\canon)}")
        axs[3].set_ylim([-17, 18])
        # axs[4].set_ylim([0, 1.05])
        for ax in axs[1:]:
            ax.axhline(0.0, color="k")
        if i_row < n_rows - 1:
            for ax in axs:
                ax.set_xlabel("")
                ax.set_xticks([])

        # filename = f"{dmp_args['dmp_type'].lower()}"
        # print(f"Saving to {filename}.tex")
        # tikzplotlib.save(filename+".tex", wrap=False)
        # save_plot(filename+".svg")

    filename = "dmps"
    print(f"Saving to {filename}.tex")
    tikzplotlib.save(filename + ".tex", wrap=False)

    plt.show()


if __name__ == "__main__":
    main()
