import numpy as np
from demos.python.bbo_of_dmps.arm2D.TaskSolverDmpArm2D import TaskSolverDmpArm2D
from demos.python.bbo_of_dmps.arm2D.TaskViapointArm2D import TaskViapointArm2D
from dmpbbo.bbo.updaters import UpdaterCovarAdaptation, UpdaterCovarDecay, UpdaterMean
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorLWR import FunctionApproximatorLWR
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from matplotlib import pyplot as plt


def plot_error_bar(x, data, color, ax):
    # plt.errorbar has strange bug with centering of mean. Using manual version instead.
    mean = np.mean(data)
    std = np.std(data)
    ax.plot(x, mean, "o", color=color)
    return ax.plot([x, x], [mean - std, mean + std], color=color, linewidth=2)


def get_demonstration(demo_name, **kwargs):
    if demo_name == "arm2D":
        n_dofs = kwargs.get("n_dofs", 2)
        duration = kwargs.get("duration", 1.0)
        n_samples = kwargs.get("n_dofs", 201)  # dt = 0.05

        # Prepare a minjerk trajectory in joint angle space
        angles_init = np.full(n_dofs, 0.0)
        angles_goal = np.full(n_dofs, np.pi / n_dofs)
        angles_goal[0] *= 0.5
        ts = np.linspace(0, duration, n_samples)
        angles_min_jerk = Trajectory.from_min_jerk(ts, angles_init, angles_goal)
        return angles_min_jerk

    elif demo_name in ["stulp13learning_meka", "stulp09compact"]:
        n = kwargs.get("traj_number", 2)
        return Trajectory.loadtxt(f"data/{demo_name}/traj{n:03}.txt")

    elif demo_name == "deprecated":
        # Train a DMP with a trajectory
        y_first = np.array([0.0])
        y_last = np.array([1.0])
        traj_demo = Trajectory.from_min_jerk(np.linspace(0, 1.0, 101), y_first, y_last)
        traj_end = Trajectory.from_min_jerk(np.linspace(0, 0.25, 26), y_last, y_last)
        traj_demo.append(traj_end)
        return traj_demo

    else:
        raise Exception("Unknown demo_name: " + demo_name)


def get_function_apps(n_dofs, fa_name, n_basis=10):
    function_apps = []
    for i_dof in range(n_dofs):
        if fa_name == "RBFN":
            function_apps.append(FunctionApproximatorRBFN(n_basis, 0.7))
        elif fa_name == "LWR":
            function_apps.append(FunctionApproximatorLWR(n_basis, 0.5))
        else:
            raise Exception("Unknown function approximator name: " + fa_name)
    return function_apps


def get_dmp(n_dofs=7, **kwargs):
    duration = kwargs.get("duration", 0.5)
    n_basis = kwargs.get("n_basis", 10)
    intersection_height = kwargs.get("intersection_height", 0.9)
    fa_name = kwargs.get("fa_name", "RBFN")
    dmp_type = kwargs.get("dmp_type", "KULVICIUS_2012_JOINING")

    # Prepare a minjerk trajectory in joint angle space
    angles_min_jerk = get_demonstration(n_dofs, duration=duration)
    link_lengths = np.full(n_dofs, 1.0 / n_dofs)

    # Prepare the function approximators
    function_apps = []
    for i_dof in range(n_dofs):
        if fa_name == "RBFN":
            fa = FunctionApproximatorRBFN(n_basis, intersection_height)
        elif fa_name == "LWR":
            fa = FunctionApproximatorLWR(n_basis, intersection_height)
        else:
            raise Exception("Unknown function approximator name: " + fa_name)
        function_apps.append(fa)

    # Train the DMP with the minjerk trajectory
    demonstration = angles_min_jerk
    dmp = Dmp.from_traj(demonstration, function_apps, **kwargs)

    return demonstration, dmp


def mae_demonstration_reproduced(traj_demonstrated, dmp):
    ts = traj_demonstrated.ts
    xs, xds, _, _ = dmp.analytical_solution(ts)
    traj_reproduced = dmp.states_as_trajectory(ts, xs, xds)
    return np.mean(np.abs(traj_demonstrated.ys - traj_reproduced.ys))


def get_task_task_solver(dmp):
    # Make task solver, based on a Dmp
    dt = 0.01
    integrate_dmp_beyond_tau_factor = 1.5
    task_solver = TaskSolverDmpArm2D(dmp, dt, integrate_dmp_beyond_tau_factor)

    # Make the task
    n_dims = 2
    viapoint = np.full(n_dims, 0.5)

    duration = dmp.tau
    n_dofs = dmp.dim_dmp()
    task = TaskViapointArm2D(
        n_dofs,
        viapoint,
        plot_arm=True,
        viapoint_time=0.6 * duration,
        viapoint_weight=1.0,
        acceleration_weight=0.0001,
        regularization_weight=0.0,
    )
    return task, task_solver


def get_updater(covar_update="decay"):
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
