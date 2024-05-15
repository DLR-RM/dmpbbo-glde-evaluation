""" Module to load the coathanger data. """
import numpy as np
from dmpbbo.dmps.Trajectory import Trajectory
from matplotlib import pyplot as plt


def compute_task_params(traj):
    """
    Compute the task parameter from a trajectory

    @param traj: The trajectory
    @return: The task parameter (i.e. the end-eff pos when the "z" coordinate is the largest.
    """
    z = 2
    index_max = np.argmax(traj.ys[:, z])
    return traj.ys[index_max, :], index_max


def plot_traj(traj, axs=None, task_params_xyz=None):
    """
    Plot a coathanger trajectory.

    @param traj: The trajectory to plot.
    @param axs: The axes to plot on.
    @param task_params_xyz: The task parameters (see compute_task_parameters)
    @return: The handles and axes
    """
    if axs is None:
        _, axs = plt.subplots(2, 3)
        axs = axs.flatten()
    if task_params_xyz is None:
        task_params_xyz, _ = compute_task_params(traj)
    hs, _ = traj.plot(axs=axs[:3])
    h = axs[3].plot(traj.ys[:, 0], traj.ys[:, 1])
    hs.append(h)
    axs[3].plot(task_params_xyz[0], task_params_xyz[1], "or")
    h = axs[4].plot(traj.ys[:, 1], traj.ys[:, 2])
    hs.append(h)
    axs[4].plot(task_params_xyz[1], task_params_xyz[2], "or")
    for ax in axs[3:]:
        ax.set_aspect("equal")
    return hs, axs


def load_data_coathanger(i_batch, n_contexts=7, n_dims=3, axs=None):
    """
    Load trajectories from the coathanger dataset.

    @param i_batch: The batch to load from (integer)
    @param n_contexts: The number of contexts (max is 7)
    @param n_dims: The number of dimensions to include in the trajectory (1,2, or 3)
    @param axs: The axes to optionally plot the trajectory on.
    @return: A list of task parameters and the corresponding trajectories.
    """
    max_n_contexts = 7
    indices = [int(x) for x in np.round(np.linspace(0, max_n_contexts - 1, n_contexts))]

    params_and_trajs = []
    cmap = plt.cm.get_cmap("copper")
    for context in indices:
        filename = f"data/coathanger23/traj_batch{i_batch}_context{context:02d}.txt"
        traj = Trajectory.loadtxt(filename)
        if n_dims < traj.dim:
            traj = Trajectory(
                traj.ts, traj.ys[:, :n_dims], traj.yds[:, :n_dims], traj.ydds[:, :n_dims]
            )

        task_params_xyz, _ = compute_task_params(traj)
        task_params = task_params_xyz[0]
        params_and_trajs.append((task_params, traj))

        # Plotting
        if axs is not None:
            hs, axs = plot_traj(traj, axs=axs, task_params_xyz=task_params_xyz)
            color = cmap(context / n_contexts)
            plt.setp(hs, color=color)

    return params_and_trajs


def main():
    """
    Main function to test the module.
    """
    for i_batch in range(4):
        _, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = axs.flatten()
        load_data_coathanger(i_batch, n_contexts=4, n_dims=3, axs=axs)
        plt.gcf().canvas.set_window_title(f"batch {i_batch}")

    plt.show()


if __name__ == "__main__":
    main()
