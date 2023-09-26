import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.dmps.Trajectory import Trajectory


def compute_task_params(traj):
    index = np.argmax(traj.ys[:, 2])
    return traj.ys[index, :], index


def plot_traj(traj, axs=None, task_params_xyz=None):
    if axs is None:
        _, axs = plt.subplots(2, 3)
        axs = axs.flatten()
    if task_params_xyz is None:
        task_params_xyz, _ = compute_task_params(traj)
    hs, _ = traj.plot(axs=axs[:3])
    h = axs[3].plot(traj.ys[:, 0], traj.ys[:, 1])
    hs.append(h)
    axs[3].plot(task_params_xyz[0], task_params_xyz[1], 'or')
    h = axs[4].plot(traj.ys[:, 1], traj.ys[:, 2])
    hs.append(h)
    axs[4].plot(task_params_xyz[1], task_params_xyz[2], 'or')
    for ax in axs[3:]:
        ax.set_aspect('equal')
    return hs, axs


def load_data_coathanger(i_batch, n_contexts=7, n_dims=3, axs=None):
    max_n_contexts = 7
    indices = [int(x) for x in np.round(np.linspace(0, max_n_contexts - 1, n_contexts))]

    params_and_trajs = []
    cmap = plt.cm.get_cmap('copper')
    for context in indices:
        filename = f"data/coathanger23/traj_batch{i_batch}_context{context:02d}.txt"
        traj = Trajectory.loadtxt(filename)
        if n_dims < traj.dim:
            traj = Trajectory(traj.ts, traj.ys[:, :n_dims], traj.yds[:, :n_dims], traj.ydds[:, :n_dims])

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
    for i_batch in range(4):
        _, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = axs.flatten()
        load_data_coathanger(i_batch, n_contexts=4, n_dims=3, axs=axs)
        plt.gcf().canvas.set_window_title(f"batch {i_batch}")

    plt.show()


if __name__ == "__main__":
    main()
