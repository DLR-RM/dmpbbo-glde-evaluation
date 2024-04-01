import numpy as np
from dmpbbo.dmps.Trajectory import Trajectory
from matplotlib import cm
from matplotlib import pyplot as plt


def get_params_and_trajs(n_trajs=5, **kwargs):
    plot_trajectories = kwargs.get("plot_trajectories", False)

    tau = 0.5
    n_time_steps = 101
    ts = np.linspace(0, tau, n_time_steps)

    n_dims = 2
    y_init = np.linspace(0.0, 0.5, n_dims)
    y_attr = np.linspace(1.0, 1.5, n_dims)

    params_train = np.linspace(-0.3, 0.3, n_trajs)
    params_and_trajs = []
    axs = None
    if plot_trajectories:
        fig, axs = plt.subplots(1, 4)
        fig.set_size_inches(4 * 4, 4 * 1)
        axs = list(axs)

    for param in params_train:
        viapoint_time = 0.5 * ts[-1]
        y_yd_ydd_viapoint = np.zeros((3 * n_dims))
        y_yd_ydd_viapoint[0] = y_init[0] + 0.2 * param
        y_yd_ydd_viapoint[1] = y_attr[1] - param
        y_yd_ydd_viapoint[2:4] = 0.1  # velocity

        traj = Trajectory.from_viapoint_polynomial(
            ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr
        )

        z = np.zeros((n_dims,))
        traj_end = Trajectory.from_polynomial(ts[:15], y_attr, z, z, y_attr, z, z)
        # traj.append(traj_end)

        cmap = cm.copper
        params_and_trajs.append((param, traj))
        if plot_trajectories:
            scaled = (param - np.min(params_train)) / (np.max(params_train) - np.min(params_train))
            scaled = np.clip(scaled, 0, 1)
            color = cmap(scaled)

            axs[0].plot(traj.ys[:, 0], traj.ys[:, 1], color=color)

            h, ax = traj.plot(axs=axs[1:])
            plt.setp(h, color=color)

    if plot_trajectories:
        axs[0].set_xlabel("y_0")
        axs[0].set_ylabel("y_1")
        axs[0].axis("equal")

    return params_and_trajs


def main():
    get_params_and_trajs(plot_trajectories=True)
    plt.show()


if __name__ == "__main__":
    main()
