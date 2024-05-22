""" Module to run a demo of the optimization of the dyn-sys parameters of the DMP. """
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo.DistributionGaussianBounded import DistributionGaussianBounded
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.Task import Task
from dmpbbo.bbo_of_dmps.TaskSolver import TaskSolver
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from utils import get_demonstration


class TaskFitTrajectory(Task):
    def __init__(self, traj_y_final, traj_length, cost_goal_weight=10000.0, **kwargs):
        self.traj_y_final = traj_y_final
        self.traj_length = traj_length
        self.cost_goal_weight = cost_goal_weight
        self.plot_trajectories = kwargs.get("plot_trajectories", [])
        if not isinstance(self.plot_trajectories, list):
            self.plot_trajectories = [self.plot_trajectories]

    def get_cost_labels(self):
        return ["accelerations", "goal after tau"]

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_of_dmps".
        @param sample: The sample from which the rollout was generated. Passing this to the cost
            function is useful when performing regularization on the sample.
        @return: costs The scalar cost components for the sample. The first item costs[0] should
            contain the total cost.
        """

        n_dims = len(self.traj_y_final)

        # Targets for function approximation
        targets = cost_vars[: self.traj_length, -n_dims:]
        diff = np.mean(np.abs(targets))

        # diff = None
        # if cost_type == "y":
        #    y_demo = self.traj_demonstrated.ys
        #    y_sample = cost_vars[:length, 1 + 0 * n_dims : 1 + 1 * n_dims]
        #    diff = np.mean(np.abs(y_demo - y_sample))
        # elif self.cost_type == "ydd":
        #    ydd_demo = self.traj_demonstrated.ydds
        #    ydd_sample = cost_vars[:length, 1 + 2 * n_dims : 1 + 3 * n_dims]
        #    diff = np.mean(np.abs(ydd_demo - ydd_sample))
        # elif self.cost_type == "targets":
        #    # Targets for function approximation
        #    targets = cost_vars[:length, -n_dims:]
        #    diff = np.mean(np.abs(targets))
        # else:
        #    raise Exception("Unknown cost type: " + self.cost_type)

        # Compute difference between goal and values of y after tau
        y_after_tau = cost_vars[
                      self.traj_length:, 1: 1 + n_dims
                      ]  # First index is time, so start at 1

        # Repeat the goal
        n = y_after_tau.shape[0]
        y_goal = self.traj_y_final
        y_goal_rep = np.tile(np.atleast_2d(y_goal).transpose(), n).transpose()

        # Compute differences
        cost_diff_goal = 0.0
        debug_plotting = False  # random.random() > 0.9
        if self.cost_goal_weight > 0.0 or debug_plotting:
            diff_goal = np.mean(np.abs(y_after_tau - y_goal_rep))
            epsilon = 0.001
            cost_diff_goal = (
                0.0 if diff_goal < epsilon else self.cost_goal_weight * (diff_goal - epsilon)
            )
            if debug_plotting:
                print(f"{diff_goal} => ({self.cost_goal_weight}) => {cost_diff_goal}")
                # plt.plot(self.traj_demonstrated.ys)
                length = self.traj_length
                plt.plot(cost_vars[:length, 1: 1 + n_dims])
                plt.plot(range(length, length + n), y_goal_rep)
                plt.plot(range(length, length + n), y_after_tau)
                plt.plot(range(length, length + n), np.abs(y_after_tau - y_goal_rep))
                plt.plot([0, length + n // 2], [diff_goal, diff_goal], "o-")
                plt.xlim([0, length + n])
                plt.show()

        costs = [diff + cost_diff_goal, diff, cost_diff_goal]
        return costs

    def plot_rollout(self, cost_vars, ax=None):
        """Plot a rollout (the cost-relevant variables).

        @param cost_vars: Rollout to plot
        @param ax: Axis to plot on (default: None, then a new axis a created)
        @return: line handles and axis
        """
        if not ax:
            ax = plt.axes()

        n_dims = len(self.traj_y_final)
        ts_sample = cost_vars[:, 0]
        offset = 1  # Set to 1 / 2 to plot vel / acc respectively
        sample = cost_vars[:, 1 + offset * n_dims: 1 + (offset + 1) * n_dims]
        if n_dims == 1 or offset > 0:
            lines_rep = ax.plot(ts_sample, sample, label="reproduced")
        else:
            lines_rep = ax.plot(sample[:, 0], sample[:, 1], label="reproduced")
        plt.setp(lines_rep, linestyle="-", linewidth=1)

        for traj in self.plot_trajectories:
            if n_dims == 1 or offset > 0:
                y = {0: traj.ys, 1: traj.yds, 2: traj.ydds}[offset]
                lines_dem = ax.plot(traj.ts, y, label="demonstration")
                ax.set_xlim([np.min(ts_sample), np.max(ts_sample)])
            else:
                lines_dem = ax.plot(traj.ys[:, 0], traj.ys[:, 1], label="demonstration")
            # plt.setp(lines_dem, linestyle=":", linewidth=4, color="#777777")
            plt.setp(lines_dem, color="#cad55c", linestyle="-", linewidth=6.0, alpha=0.8, zorder=1)

        labels = [r"$y~(m)$", r"$\dot{y}~(m/s)$", r"$\ddot{y}~(m/s^2)$"]
        ax.set_xlabel(r"$time (s)$")
        ax.set_ylabel(labels[offset])

        return lines_rep, ax


class TaskSolverDmpDynSys(TaskSolver):
    def __init__(self, trajectory, dmp_type="KULVICIUS_2012_JOINING", decoupled=False):
        self._trajectory = trajectory
        self._dmp_type = dmp_type
        self._decoupled = decoupled

        # Compute the time steps with which to integrate
        dur = self._trajectory.duration
        dur_integrate = 1.25 * dur
        dt = dur / (self._trajectory.length - 1)
        self._ts = np.append(self._trajectory.ts, np.arange(dur + dt, dur_integrate, dt))

        self._dmp_type = dmp_type
        function_apps = None
        dmp = Dmp.from_traj(trajectory, function_apps, dmp_type=dmp_type, save_training_data=True)
        if decoupled:
            dmp.decouple_parameters()

        init_values = {"spring_system": {}}

        # This holds for all systems
        init_values["spring_system"]["spring_constant"] = dmp._spring_system.spring_constant

        if dmp_type == "IJSPEERT_2002_MOVEMENT":
            pass

        elif dmp_type == "KULVICIUS_2012_JOINING":
            init_values["goal_system"] = {"alpha": dmp._goal_system.alpha}

        elif "2022" in dmp_type or "SCT23" in dmp_type:
            init_values["spring_system"][
                "damping_coefficient"
            ] = dmp._spring_system.damping_coefficient
            init_values["goal_system"] = {
                "alpha": dmp._goal_system.alpha,
                "t_infl_ratio": dmp._goal_system.t_infl_ratio,
                "v": dmp._goal_system.v,
            }
            if "DAMPING" in dmp_type:
                init_values["damping_system"] = {"alpha": dmp._damping_system.alpha}
        else:
            raise ValueError(f"Unknown dmp type: {dmp_type}")

        self._init_values = init_values

        self._dmp = dmp

    @staticmethod
    def _params_dict_to_array(params_dict):
        params_list = []
        for system_name in params_dict:
            for param_name, values in params_dict[system_name].items():
                if np.isscalar(values):
                    params_list.append(values)
                else:
                    params_list.extend(values)
        return np.asarray(params_list)

    def suggest_distribution_init(self, covar_scale=0.25):
        """
        Compute the initial distribution from the _init_values member.

        @param covar_scale: The covariance diagonal value, i.e. scale*range for the range in _init_values.
        @return: The initialized distribution.
        """
        mean_init = self._params_dict_to_array(self._init_values)

        min_values = {}  # noqa
        max_values = {}  # noqa
        covar_init_dict = {}  # noqa
        for system in self._init_values:
            min_values[system] = {key: 0.1 * val for key, val in self._init_values[system].items()}
            max_values[system] = {key: 2.0 * val for key, val in self._init_values[system].items()}
            covars = {
                key: np.square(covar_scale * p) for key, p in self._init_values[system].items()
            }
            if not self._decoupled:
                covars = {key: np.mean(vals) for key, vals in covars.items()}
            covar_init_dict[system] = covars

        if self._dmp_type == "2022":
            if self._decoupled:
                covar_init_dict["goal_system"]["t_infl_ratio"][:] = covar_scale * covar_scale
                min_values["goal_system"]["v"][:] = 1.0
                max_values["goal_system"]["v"][:] = 8.0
                min_values["goal_system"]["t_infl_ratio"][:] = 0.0
                max_values["goal_system"]["t_infl_ratio"][:] = 0.9
            else:
                covar_init_dict["goal_system"]["t_infl_ratio"] = covar_scale * covar_scale
                min_values["goal_system"]["v"] = 1.0
                max_values["goal_system"]["v"] = 8.0
                min_values["goal_system"]["t_infl_ratio"] = 0.0
                max_values["goal_system"]["t_infl_ratio"] = 0.9

        lower_bound = self._params_dict_to_array(min_values)
        upper_bound = self._params_dict_to_array(max_values)
        covar_diag_init = self._params_dict_to_array(covar_init_dict)

        distribution = DistributionGaussianBounded(
            mean_init, np.diag(covar_diag_init), lower_bound, upper_bound
        )
        return distribution

    def get_dmp(self, sample):
        """
        Generate a DMP, given a sample from the distribution

        @param sample: The sample from the distribution
        @return: The DMP, in which the dynsys parameters have been initialized with the values in the sample.
        """
        params_cur = {}
        offset = 0
        for system_name in self._init_values:
            params_cur[system_name] = {}
            for param_name in self._init_values[system_name]:
                init_param_vals = self._init_values[system_name][param_name]
                if np.isscalar(init_param_vals):
                    cur_val = sample[offset]
                    offset += 1
                else:
                    n = len(init_param_vals)
                    cur_val = sample[offset: offset + n]
                    offset += n
                params_cur[system_name][param_name] = cur_val

        self._dmp.set_params_dyn_sys(params_cur)
        if self._dmp_type != "2022":
            # Ensure critical damping
            self._dmp._spring_system.damping_coefficient = np.sqrt(
                4.0 * self._dmp._spring_system.spring_constant
            )
        return self._dmp

    def perform_rollout(self, sample, **kwargs):
        self._dmp = self.get_dmp(sample)

        xs, xds, _, _ = self._dmp.analytical_solution(self._ts)
        traj = self._dmp.states_as_trajectory(self._ts, xs, xds)

        # Add targets
        inputs_phase, targets = self._dmp._compute_targets(self._trajectory)

        # Integration time is longer that trajectory duration
        # targets is for trajectory duration only, so pad with zeros
        append_zeros = np.zeros((traj.length - targets.shape[0], targets.shape[1]))
        traj.misc = np.concatenate((targets, append_zeros))

        cost_vars = traj.as_matrix()

        return cost_vars


def plot_before_after(session, traj, directory=None):
    """
    Plot the DMP before and after optimization

    @param session: The learning session from which to plot from.
    @param traj: The demonstrated trajectory.
    @param directory: The directory to which to save the results to.
    """
    colors = {"before": "red", "after": "green"}

    axs_dmp = None

    n_updates = session.get_n_updates()
    for label, i_update in {"before": 0, "after": n_updates - 1}.items():
        distribution = session.ask("distribution", i_update)
        print(distribution.mean)
        task_solver = session.ask("task_solver")
        dmp = task_solver.get_dmp(distribution.mean)

        for fa in ["without", "with"]:
            if fa == "with":
                # Train the dmp with function approximators

                fas = [FunctionApproximatorRBFN(30, 0.85) for _ in range(dmp.dim_dmp())]
                dmp.train(traj, function_approximators=fas)
                if directory:
                    # Save the parameter vector
                    dmp.set_selected_param_names("weights")
                    fa_params = dmp.get_param_vector()
                    session.save(fa_params, directory, f"fa_params_{label}")

            dmp_name = f"dmp_{label}_{fa}_fa"
            if directory:
                session.save(dmp, directory, dmp_name)

            plot_traj = traj if fa == "without" else None
            h, axs_dmp = dmp.plot(
                axs=axs_dmp,
                plot_demonstration=plot_traj,
                plot_no_forcing_term_also=True,
                plot_compact=True,
            )
            plt.setp(h, color=colors[label])
            if fa == "without":
                plt.setp(h, linestyle="--")


def get_dmp_before_after(session):
    """
    Get the DMP before and after optimization

    @param session: The learning session from which to get the DMPs.
    @return: A list with 2 dmps (before and after optimization)
    """
    dmps = {}
    n_updates = session.get_n_updates()
    for label, i_update in {"before": 0, "after": n_updates - 1}.items():
        distribution = session.ask("distribution", i_update)
        task_solver = session.ask("task_solver")
        dmp = task_solver.get_dmp(distribution.mean)
        dmps[label] = dmp
    return dmps


def main(directory=None):
    """
    Main script.

    @param directory: Directory to save results to.
    """
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=300)

    # ts = np.linspace(0.0, 1.0, 101)
    # traj_demo = Trajectory.from_min_jerk(ts, np.array([0.5]), np.array([1.5]))

    demo_name = "stulp09compact"
    # demo_name = "stulp13learning_meka"
    traj_number = 7
    traj_demo = get_demonstration(demo_name, traj_number=traj_number)
    i_dim = 1
    traj_demo = Trajectory(
        traj_demo.ts, traj_demo.ys[:, i_dim], traj_demo.yds[:, i_dim], traj_demo.ydds[:, i_dim]
    )

    # Create task
    goal_cost_weight = 100.0
    traj_y_final = traj_demo.y_final
    traj_length = traj_demo.length
    task = TaskFitTrajectory(
        traj_y_final, traj_length, goal_cost_weight, plot_trajectories=traj_demo
    )

    # Create task solver
    dmp_type = "2022"
    task_solver = TaskSolverDmpDynSys(traj_demo, dmp_type)

    # Run the optimization
    distribution = task_solver.suggest_distribution_init()
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.97)
    n_updates = 50
    n_samples_per_update = 20
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update
    )

    session.plot()

    if directory:
        session.save_all(Path(directory, dmp_type))

    plot_before_after(session, traj_demo)

    plt.show()


if __name__ == "__main__":
    main()
