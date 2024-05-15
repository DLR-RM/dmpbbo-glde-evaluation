""" Module with several utility functions. """
import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorLWR import FunctionApproximatorLWR
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN


def plot_error_bar(x, data, color, ax):
    """
    Plot an error bar on an axis.

    @param x: The value on the x-axis at which to plot the bat
    @param data:  The data for which to plot the bar.
    @param color:  The color of the bar.
    @param ax: The axis on which to plot.
    @return: A handle to the bar.
    """
    # plt.errorbar has strange bug with centering of mean. Using manual version instead.
    mean = np.mean(data)
    std = np.std(data)
    ax.plot(x, mean, "o", color=color)
    return ax.plot([x, x], [mean - std, mean + std], color=color, linewidth=2)


def get_demonstration(dataset_name, traj_number=2):
    """
    Get a demonstration for a certain dataset

    @param dataset_name: The name of the dataset from which to get the demo.
    @param traj_number: Which trajectory to return (int)
    @return:
    """
    if dataset_name in ["stulp13learning_meka", "stulp09compact"]:
        return Trajectory.loadtxt(f"data/{dataset_name}/traj{traj_number:03}.txt")

    else:
        raise Exception("Unknown dataset name: " + dataset_name)


def get_function_apps(n_dofs, fa_name, n_basis=10):
    """
    Convenience function to initialize a list of function approximators.

    @param n_dofs: The number of function approximators in the list.
    @param fa_name: The name of the function approximator ("RBFN" or "LWR")
    @param n_basis: The number of basis functions.
    @return: A list of initialized (but untrained) function approximators.
    """
    function_apps = []
    for i_dof in range(n_dofs):
        if fa_name == "RBFN":
            function_apps.append(FunctionApproximatorRBFN(n_basis, 0.7))
        elif fa_name == "LWR":
            function_apps.append(FunctionApproximatorLWR(n_basis, 0.5))
        else:
            raise Exception("Unknown function approximator name: " + fa_name)
    return function_apps

def mae_demonstration_reproduced(traj_demonstrated, dmp):
    """
    Get the mean absolute error between a demonstrated and reproduced trajectory.

    @param traj_demonstrated: The demonstrated trajectory.
    @param dmp: The DMP that will reproduce the trajectory.
    @return: The mean absolute error between the demonstrated and reproduced trajectory.
    """
    ts = traj_demonstrated.ts
    xs, xds, _, _ = dmp.analytical_solution(ts)
    traj_reproduced = dmp.states_as_trajectory(ts, xs, xds)
    return np.mean(np.abs(traj_demonstrated.ys - traj_reproduced.ys))


def get_updater(covar_update="decay"):
    """
    Get the updater for the optimization process.

    @param covar_update: Name of the updater ("none" (only mean, no covar update), "decay", "CMA")
    @return: The updater (inherits from dmpbbo.bbo.updaters.Updater)
    """
    if covar_update == "none":
        updater = UpdaterMean(eliteness=10, weighting_method="PI-BB")
    elif covar_update == "decay":
        updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.9)
    else:
        updater = UpdaterCovarAdaptation(
            eliteness=10,
            weighting_method="PI-BB",
            max_level=None,
            min_level=1.0,
            diag_only=False,
            learning_rate=0.5,
        )
    return updater


def plot_dmp_parameterization(demonstration, ts, dmp, axs, plot_without_forcing=False):
    """
    Plot the DMP and the parameters of the function approximator

    @param demonstration: The demonstrated trajectory
    @param ts: The time stess
    @param dmp: The DMP to generate the reproduced trajectory
    @param axs: The axs to plot on (list of 6 axes)
    @param plot_without_forcing: Whether to plot without forcing term or not. Default = False
    """
    # Comparison between demonstration and DMP reproduction
    dmp.plot_comparison(demonstration, ts=ts, axs=axs[0:3])

    if plot_without_forcing:
        # Plot trajectory without forcing term also
        params = dmp.get_param_vector()
        dmp.set_param_vector(np.zeros(params.shape))
        xs, xds, _, _ = dmp.analytical_solution(ts)
        traj_reproduced_no_fa = dmp.states_as_trajectory(ts, xs, xds)
        h_no_fa, _ = traj_reproduced_no_fa.plot(axs[0:3])
        plt.setp(h_no_fa, linestyle="-", linewidth=1, color=(0.7, 0.3, 0.3))
        plt.setp(h_no_fa, label="reproduced (no forcing)")
        dmp.set_param_vector(params)

    for fa in dmp._function_approximators:  # noqa
        fa.plot(ax=axs[3], plot_residuals=False, plot_model_parameters=False)
        axs[3].set_xlabel("phase")

        values = fa.get_param_vector()
        axs[4].plot(values, ".")

        axs[5].hist(values, orientation="horizontal")

    for ax in axs[3:6]:
        ax.invert_xaxis()  # Because phase goes from 1 to 0
        ax.plot(ax.get_xlim(), [0, 0], "-k")
