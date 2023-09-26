import copy
import pprint
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo import json_for_cpp
from dmpbbo.bbo.updaters import UpdaterCovarDecay
from dmpbbo.bbo_of_dmps.Task import Task
from dmpbbo.bbo_of_dmps.TaskSolver import TaskSolver
from dmpbbo.bbo_of_dmps.run_optimization_task import run_optimization_task
from dmpbbo.dmps.DmpContextualTwoStep import DmpContextualTwoStep
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorGPR import FunctionApproximatorGPR
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.json_for_cpp import savejson, loadjson
from dmpbbo_sct_experiments.demo_optimize_dyn_sys_parameters import (TaskFitTrajectory, TaskSolverDmpDynSys,
                                                          get_dmp_before_after)
from dmpbbo_sct_experiments.load_data_coathanger import load_data_coathanger, plot_traj, compute_task_params
from dmpbbo_sct_experiments.utils import plot_error_bar


class TaskFitMultiTrajectory(Task):
    """
    Fit multiple trajectories at the same time, the cost being the sum of costs over all trajectories.

    Delegates most function calls to TaskFitTrajectory, i.e. one call per trajectory.
    """

    def __init__(self, trajectories, cost_goal_weight=10000.0):
        self.task_fit_trajectory = []
        for traj in trajectories:
            task = TaskFitTrajectory(traj.y_final, traj.length, cost_goal_weight)
            self.task_fit_trajectory.append(task)

    def get_cost_labels(self):
        self.task_fit_trajectory[0].get_cost_labels()

    def _get_cost_vars_list(self, cost_vars):
        cost_vars_list = []
        for i_traj in range(len(self.task_fit_trajectory)):
            cur_cost_vars = cost_vars[cost_vars[:, -1] == i_traj, :]
            cur_cost_vars = cur_cost_vars[:, :-1]
            cost_vars_list.append(cur_cost_vars)
        return cost_vars_list

    def evaluate_rollout(self, cost_vars, sample):
        cost_vars_list = self._get_cost_vars_list(cost_vars)
        costs = np.zeros((3,))
        for cur_cost_vars, task in zip(cost_vars_list, self.task_fit_trajectory):
            cur_costs = task.evaluate_rollout(cur_cost_vars, sample)
            costs = [costs[i] + cur_costs[i] for i in range(len(costs))]
        return costs

    def plot_rollout(self, cost_vars, ax=None):
        cost_vars_list = self._get_cost_vars_list(cost_vars)
        lines = []
        for cur_cost_vars, task in zip(cost_vars_list, self.task_fit_trajectory):
            cur_lines, ax = task.plot_rollout(cur_cost_vars, ax=ax)
            lines.extend(lines)
        return lines, ax


class TaskSolverDmpDynSysMultiTraj(TaskSolver):
    def __init__(self, trajs, dmp_type="KULVICIUS_2012_JOINING", decoupled=False):
        self.trajs = trajs

        # Get the time steps with which to integrate (i.e. longest duration + n% beyond tau)
        max_duration = 0.0
        for traj in trajs:
            max_duration = max(traj.duration, max_duration)
        self.ts_integrate = np.arange(0.0, 1.25 * max_duration, trajs[0]._dt_mean)

        train_traj = trajs[len(trajs) // 2]
        self.task_solver = TaskSolverDmpDynSys(train_traj, dmp_type, decoupled)

    def suggest_distribution_init(self):
        return self.task_solver.suggest_distribution_init()

    def get_dmp(self, sample):
        return self.task_solver.get_dmp(sample)

    def perform_rollout(self, sample, **kwargs):

        dmp = self.task_solver.get_dmp(sample)

        # For each traj
        #   Integrate DMP a bit longer than traj length beyond tau
        #   Compute targets for traj length before tau (and pad with zeros)
        #   Add traj number in last column
        #   Concatenate
        #

        # 1+ => time
        # 3 + 1 =>  y,yd,ydd + target
        # +1 => number for trajectory
        n_cost_vars = 1 + (dmp.dim_y * (3 + 1)) + 1
        cost_vars = np.empty((0, n_cost_vars))
        for i_traj, demo_traj in enumerate(self.trajs):
            # Get DMP trajectory
            dmp.tau = demo_traj.duration
            xs, xds, _, _ = dmp.analytical_solution(self.ts_integrate)
            dmp_traj = dmp.states_as_trajectory(self.ts_integrate, xs, xds)

            # Compute targets for the trajectory
            # Will be of length le(demo_traj.length), not len (ts_beyond_tau)
            _, targets = dmp._compute_targets(demo_traj)

            # Integration time is longer that trajectory duration
            # targets is for trajectory duration only, so pad with zeros
            n_tau = demo_traj.length
            n_beyond_tau = len(self.ts_integrate)
            append_zeros = np.zeros((n_beyond_tau - n_tau, targets.shape[1]))
            dmp_traj.misc = np.concatenate((targets, append_zeros))

            cur_cost_vars = dmp_traj.as_matrix()

            # Add a column with the number of the trajectory
            traj_number_column = np.full((cur_cost_vars.shape[0], 1), i_traj)
            cur_cost_vars = np.append(cur_cost_vars, traj_number_column, axis=1)

            cost_vars = np.concatenate((cost_vars, cur_cost_vars))

        return cost_vars


def run_optimization(
        trajs,
        dmp_type,
        decoupled,
        n_updates=20,
        n_samples_per_update=10
):
    """
    Optimize the dyn_sys parameters for a given set of trajectories.

    @param trajs: The trajectories for training
    @param dmp_type: The DMP type, e.g. "IJSPEERT_2002_MOVEMENT"
    @param decoupled: Whether the dyn_sys parameters are decoupled or not
    @param n_updates: Number of updates for the optimization
    @param n_samples_per_update: Number of samples per optimization update
    @return: A dict with two dmps, one before optimization, and one after the optimization.
    """
    goal_cost_weight = 100.0
    task = TaskFitMultiTrajectory(trajs, goal_cost_weight)
    task_solver = TaskSolverDmpDynSysMultiTraj(trajs, dmp_type, decoupled)
    distribution = task_solver.suggest_distribution_init()
    updater = UpdaterCovarDecay(eliteness=10, weighting_method="PI-BB", covar_decay_factor=0.95)
    session = run_optimization_task(
        task, task_solver, distribution, updater, n_updates, n_samples_per_update
    )
    dmps = get_dmp_before_after(session)

    return dmps, session


def run_optimization_cached(trajs, dmp_type, decoupled, directory, n_updates=50, overwrite=False, plot=False):
    """
    Optimize the dyn_sys parameters for a given set of trajectories, or load it from file if it is already available.

    @param trajs: The trajectories for training
    @param dmp_type: The DMP type, e.g. "IJSPEERT_2002_MOVEMENT"
    @param decoupled: Whether the dyn_sys parameters are decoupled or not
    @param directory: Directory for reading/writing
    @param n_updates: Number of updates for the optimization
    @param overwrite: Whether to re-optimize in any case and overwrite existing dmp files, if they exist.
    @param plot: Whether to plot the optimization, and save it to PNG
    @return: A dict with two dmps, one before optimization, and one after the optimization.
    """
    n_samples = 10

    exp_key = to_key(dmp_type, decoupled, len(trajs))
    basename = Path(directory, exp_key)
    filenames_dict = {ba: f"{basename}_dmp_{ba}.json" for ba in ["before", "after"]}

    # Try to read file if it exists (and we are not overwriting)
    if Path(filenames_dict["after"]).exists() and not overwrite:
        dmps = {}
        for before_after, filename in filenames_dict.items():
            print(f"    loading DMP from {filename}")
            dmps[before_after] = json_for_cpp.loadjson(filename)
        return dmps  # No need to run optimization below

    # File not there or we are overwriting existing results: run optimization!
    print(f"    running optimization")
    dmps, session = run_optimization(
        trajs,
        dmp_type,
        decoupled,
        n_updates=n_updates,
        n_samples_per_update=n_samples
    )

    for before_after, filename in filenames_dict.items():
        print(f"    saving DMP to {filename}")
        json_for_cpp.savejson(filename, dmps[before_after])

    save_session = False
    if save_session:
        session_directory = directory
        session_directory = Path(session_directory, exp_key)
        print(f"    saving session to {session_directory}")
        session.save_all(session_directory)

    if plot:
        filename_plot = str(basename) + ".png"
        print(f"    saving plot to {filename_plot}")
        session.plot()
        plt.gcf().canvas.set_window_title(filename_plot)
        plt.savefig(str(filename_plot))

    return dmps


def train_contextual_dmp(params_and_trajs, dmp):
    """
    Train a contextual dmp on multiple trajectories, or load it from file if it is already available.

    @param params_and_trajs: The list of task parameters and trajectories for training
    @param dmp: The untrained contextual dmp
    @return: The trained contextual dmp
    """
    fa_ppf = FunctionApproximatorGPR(max_covariance=1.0, lengths=0.07)
    dmp_contextual = DmpContextualTwoStep.from_dmp(
        params_and_trajs, dmp, ["weights"], fa_ppf, save_training_data=True
    )
    return dmp_contextual


def train_contextual_dmp_cached(filename, params_and_trajs, dmp, overwrite=False, plot_ppf=False):
    """
    Train a contextual dmp on multiple trajectories, or load it from file if it is already available.

    @param filename: File for reading/writing
    @param params_and_trajs: The list of task parameters and trajectories for training
    @param dmp: The untrained contextual dmp
    @param overwrite: Whether to train in any case and overwrite existing dmp files, if they exist.
    @param plot_ppf: Plot the policy-parameter-function
    @return: The trained contextual dmp
    """
    if Path(filename).exists() and not overwrite:
        print(f"    {filename} already exists: loading DMP contextual")
        dmp_contextual = json_for_cpp.loadjson(filename)
        return dmp_contextual

    print(f"    training contextual DMP")
    dmp_contextual = train_contextual_dmp(params_and_trajs, dmp)

    print(f"    saving to {filename}")
    json_for_cpp.savejson(filename, dmp_contextual)

    plot_dmp_integration = False
    if plot_dmp_integration:
        trajs = [pt[1] for pt in params_and_trajs]
        _, axs = dmp_contextual.plot(
            params_and_trajs, plot_no_forcing_term_also=True, plot_demonstrations=trajs
        )
        filename_plot = str(filename).replace(".json", ".png")
        print(f"    saving plot to {filename_plot}")
        plt.gcf().canvas.set_window_title(filename_plot)
        plt.savefig(filename_plot)

    if plot_ppf:
        _, _ = dmp_contextual.plot_policy_parameter_function(plot_model_parameters=False)
        filename_plot = str(filename).replace(".json", "_ppf.png")
        print(f"    saving plot to {filename_plot}")
        plt.gcf().canvas.set_window_title(filename_plot)
        plt.savefig(filename_plot)

    return dmp_contextual


def evaluate_deprecated(dmp_contextual, params_and_trajs, tp_offset=0.0, plot_me=False, axs=None):
    # dmp_contextual.plot(params_and_trajs)
    hs = []
    maes = []
    for i_traj in range(len(params_and_trajs)):
        task_params = np.atleast_1d(params_and_trajs[i_traj][0] + tp_offset)
        traj = params_and_trajs[i_traj][1]
        ts = traj.ts
        xs, xds, forcing_terms, fa_outputs = dmp_contextual.analytical_solution(task_params, ts=ts)
        traj_reproduced = dmp_contextual.states_as_trajectory(ts, xs, xds)
        maes.append(np.mean(np.abs(traj.ys - traj_reproduced.ys)))

        if plot_me:
            h, axs = dmp_contextual.dmp.plot(ts, xs, xds, forcing_terms=forcing_terms, fa_outputs=fa_outputs, axs=axs)
            h_traj, _ = traj.plot(axs=axs[1:4])
            plt.setp(h_traj, color='k', linestyle='--')
            hs.extend(h)

    return np.mean(maes), hs, axs


def to_key(dmp_type, decoupled, n_trajs):
    return f"dmp{dmp_type}_{'de' if decoupled else ''}coupled_ntrajs{n_trajs:02d}"


def integrate_dmp_contextual(dmp_context, task_params, ts):
    """ Convenience function to integrate a dmp.
    @return: The trajectory resulting from integrating the dmp.
    """
    xs, xds, forcing_terms, fa_outputs = dmp_context.analytical_solution(task_params, ts=ts)
    traj_repro = dmp_context.states_as_trajectory(ts, xs, xds)
    return traj_repro


def evaluate_dmp_contextual_with_traj(dmp_context, traj, task_params, index, task_params_xyz=None, axs=None, color='k'):
    traj_repro = integrate_dmp_contextual(dmp_context, task_params, traj.ts)

    tp_repro = traj_repro.ys[index, 0]
    mae_tp = np.abs(tp_repro - task_params)[0]
    mean_dist_ys = np.mean(np.linalg.norm(traj_repro.ys - traj.ys, axis=1))

    if axs is not None:
        hs, axs = plot_traj(traj, axs, task_params_xyz=task_params_xyz)
        plt.setp(hs, color="#cccccc")

        hs, axs = plot_traj(traj_repro, axs)
        plt.setp(hs, color=color)

        xs_plot = [traj_repro.ys[index, 0], task_params_xyz[0]]
        ys_plot = [traj_repro.ys[index, 1], task_params_xyz[1]]
        axs[3].plot(xs_plot, ys_plot, 'o-k')

    return traj_repro, mae_tp, mean_dist_ys


def evaluate_dmp_contextual(dmp_context, params_and_trajs_all, axs=None):
    # Plot training trajectories and intermediate test trajectories
    trajs_repro = []

    diff_tp = {'train': []}
    diff_ys = {'train': []}
    for i_task_param in [0, 2, 4, 6]:
        traj = params_and_trajs_all[i_task_param][1]
        task_params_xyz, index = compute_task_params(traj)
        task_params = np.atleast_1d(params_and_trajs_all[i_task_param][0])

        traj_repro, mae_tp, mean_dist_ys = evaluate_dmp_contextual_with_traj(dmp_context, traj, task_params, index,
                                                                             task_params_xyz=task_params_xyz,
                                                                             axs=axs, color="#7777dd")
        trajs_repro.append((task_params, traj_repro))
        diff_tp['train'].append(mae_tp)
        diff_ys['train'].append(mean_dist_ys)

    diff_tp['interpolate'] = []
    diff_ys['interpolate'] = []
    for i_task_param in [1, 3, 5]:
        traj1 = params_and_trajs_all[i_task_param - 1][1]
        traj2 = params_and_trajs_all[i_task_param + 1][1]
        ys = 0.5 * (traj1.ys + traj2.ys)
        traj = Trajectory(traj1.ts, ys)
        task_params_xyz, index = compute_task_params(traj)
        task_params = np.atleast_1d(task_params_xyz[0])

        traj_repro, mae_tp, mean_dist_ys = evaluate_dmp_contextual_with_traj(dmp_context, traj, task_params, index,
                                                                             task_params_xyz=task_params_xyz,
                                                                             axs=axs, color="#dd7777")
        trajs_repro.append((task_params, traj_repro))
        diff_tp['interpolate'].append(mae_tp)
        diff_ys['interpolate'].append(mean_dist_ys)

    diff_tp['extrapolate'] = []
    diff_ys['extrapolate'] = []
    left_tp = params_and_trajs_all[0][0]
    left_traj = params_and_trajs_all[0][1]
    task_params_xyz, index = compute_task_params(left_traj)
    for task_param in [left_tp - 0.21, left_tp - 0.14, left_tp - 0.07]:

        task_params = np.atleast_1d(task_param)
        traj_repro = integrate_dmp_contextual(dmp_context, task_params, left_traj.ts)
        trajs_repro.append((task_params, traj_repro))
        mean_dist = np.mean(np.linalg.norm(traj_repro.ys - left_traj.ys, axis=1))
        diff_ys['extrapolate'].append(mean_dist)

        tp_repro = traj_repro.ys[index, 0]
        diff = np.abs(tp_repro - task_params)[0]
        diff_tp['extrapolate'].append(diff)

        if axs is not None:
            hs, axs = plot_traj(traj_repro, axs)
            plt.setp(hs, color="#77dd77")

    return trajs_repro, diff_tp, diff_ys


def add_orientation(traj):
    """
    Add a fixed orientation to the trajectory.

    In the experiment, the orientation of the end-eff remains fixed.

    @param traj: A trajectory with x,y,z of end-eff
    @return: Traj with quaternion orientation added.
    """
    # Add fixed orientation values here.
    n = traj.length
    orientations = np.array([0.019433, 0.37355, 0.92488, 0.06834])
    orientations_rep = np.tile(orientations, (n, 1))
    as_matrix = np.column_stack((traj.ts, traj.ys, orientations_rep))
    return as_matrix


def plot_results_list(diff_tp_list, diff_ys_list):
    """ Plot the results as boxplots.

    This function includes a lot of assumptions about axis limits and experiment names.
    """
    pprint.pprint(diff_ys_list)
    diff_tp_all = {exp_key: {'train': [], 'interpolate': [], 'extrapolate': []} for exp_key in diff_ys_list[0]}
    diff_ys_all = {exp_key: {'train': [], 'interpolate': [], 'extrapolate': []} for exp_key in diff_ys_list[0]}
    n_results = len(diff_ys_list)
    for i_batch in range(n_results):
        for exp_key in diff_ys_list[i_batch]:
            for ti in ['train', 'interpolate', 'extrapolate']:
                diff_ys_all[exp_key][ti].append(100 * diff_ys_list[i_batch][exp_key][ti])
                diff_tp_all[exp_key][ti].append(100 * diff_tp_list[i_batch][exp_key][ti])

    pprint.pprint(diff_ys_all)
    fig, axs = plt.subplots(1, 2)
    x = 0.0
    x_ticks = []
    x_ticklabels = []
    for dmp_type in ['IJSPEERT_2002_MOVEMENT_coupled', 'KULVICIUS_2012_JOINING_coupled', '2022_decoupled']:
        print("________________________________")
        print(dmp_type)
        before_after = 'after' if '2022' in dmp_type else 'before'
        x += 0.8
        exp_keys_found = []
        for k in diff_ys_all:
            if dmp_type in k and before_after in k:
                exp_keys_found.append(k)
        if not len(exp_keys_found) == 1:
            raise ValueError("None or multiple keys found, something is wrong.")

        exp_key = exp_keys_found[0]
        x_ticks.append(x)
        cur_label = f"{dmp_type[:4]}"
        x_ticklabels.append(cur_label)

        print(cur_label)
        print(exp_key)
        for diff, yt in zip([diff_ys_all, diff_tp_all], ['ys', 'tp']):
            train = np.mean(diff[exp_key]['train'])
            inter_ratio = np.mean(diff[exp_key]['interpolate']) / train
            extra_ratio = np.mean(diff[exp_key]['extrapolate']) / train
            print(f"  {yt}  interpolate = {inter_ratio:.2f}   extrapolate = {extra_ratio:.2f}")

        colors = {'train': 'black', 'interpolate': 'orange', 'extrapolate': 'red'}

        xs = [x + d for d in [-0.25, 0, 0.25]]
        print(cur_label)
        print(exp_key)
        means = []
        for ti, xc in zip(['train', 'interpolate', 'extrapolate'], xs):
            mean = np.mean(diff_ys_all[exp_key][ti])
            std = np.std(diff_ys_all[exp_key][ti])
            print(f"{ti} {mean} {std}")

            plot_error_bar(xc, diff_ys_all[exp_key][ti], colors[ti], axs[0])
            axs[0].axvline(x + 0.4, color="#cccccc", linewidth=1)

        means = []
        for ti, xc in zip(['train', 'interpolate', 'extrapolate'], xs):
            plot_error_bar(xc, diff_tp_all[exp_key][ti], colors[ti], axs[1])
            axs[1].axvline(x + 0.4, color="#cccccc", linewidth=1)

    #axs[0].set_ylim([0, 11])
    #axs[1].set_ylim([0, 20])
    axs[0].set_ylabel('mean diff in traj (cm)')
    axs[1].set_ylabel('diff in task parameter (cm)')
    for ax in axs:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=40, ha="right")


def plot_results(diff_tp, diff_ys):
    # pprint.pprint(diff_tp_all)
    # pprint.pprint(diff_ys_all)
    fig, axs = plt.subplots(1, 2)
    x = 0
    w = 0.2
    for exp_key in diff_tp:
        xs = [x - 1.5 * w, x, x + 1.5 * w]
        axs[0].bar(xs, [100 * v for v in diff_ys[exp_key].values()], width=w)
        axs[1].bar(xs, [100 * v for v in diff_tp[exp_key].values()], width=w)
        x += 1
    axs[0].set_ylabel('mean diff in traj (cm)')
    axs[1].set_ylabel('diff in task parameter (cm)')
    for ax in axs:
        ax.set_xticklabels(['', 'IJS', 'KUL', 'NEW'])
        # ax.set_yscale("log")


def main_with_trajs(experiments, params_and_trajs_all, main_directory, n_basis=10, save_for_robot=False):
    """
    Do overall optimization and training for a set of experiments and trajectories.

    @param experiments: List of experiment tuples with dmp_type (str) and decoupled (bool)
    @param params_and_trajs_all: List of tuples with task parameters and corresponding demonstrated trajectories
    @param main_directory: The trajectory for reading/writing results to
    @param n_basis: Number of basis functions for the dmp function approximators
    @param save_for_robot: Whether the trajectories should be saved for execution on the robot.
    @return: dict with summary of evaluation statistics
    """
    max_n_contexts = 7
    n_contexts = 4
    indices = [int(x) for x in np.round(np.linspace(0, max_n_contexts - 1, n_contexts))]
    params_and_trajs = [params_and_trajs_all[i] for i in indices]

    trajs = [pt[1] for pt in params_and_trajs]

    # Run optimizations for all experiments and gather dmps in a dict
    # Here, there are no function approximators yet
    dmps_all = {}
    for dmp_type, decoupled in experiments:
        optim_args = {"n_updates": 50, "overwrite": False, "plot": True}
        dmps = run_optimization_cached(trajs, dmp_type, decoupled, main_directory, **optim_args)
        dmps_all[to_key(dmp_type, decoupled, len(trajs))] = dmps

    n_dims = params_and_trajs[0][1].dim
    args = {"overwrite": False}
    axs_ppf = None
    fig_ppf = None
    plot_ppf = False
    if plot_ppf:
        fig_ppf, axs_ppf = plt.subplots(3, n_basis)
        axs_ppf = axs_ppf.flatten()

    diff_tp_all = {}
    diff_ys_all = {}
    for exp_key, dmp_before_after in dmps_all.items():
        for before_after, dmp in dmp_before_after.items():

            dmp = dmps_all[exp_key][before_after]
            basename = f"contextual_{exp_key}_{before_after}"

            intersection_height = 0.9
            fa_dmp = FunctionApproximatorRBFN(n_basis, intersection_height)
            fas_dmp = [copy.deepcopy(fa_dmp) for _ in range(n_dims)]
            dmp._function_approximators = fas_dmp

            file_path = Path(main_directory, f"{basename}_n_basis{n_basis:02d}.json")
            dmp_context = train_contextual_dmp_cached(file_path, params_and_trajs, dmp, **args)

            # Plot policy parameter function
            if plot_ppf:
                hs, _ = dmp_context.plot_policy_parameter_function(axs=axs_ppf)
                fig_ppf.canvas.set_window_title("PPF")
                for h in hs[0:]:
                    linestyles = {"before": "--", "after": "-"}
                    plt.setp(h, linestyle=linestyles[before_after])
                for ax in axs_ppf:
                    ax.axhline(y=0, color='r')
                plt.savefig(f"{file_path}.png")

            plot_trajs = False
            if plot_trajs:
                fig, axs = plt.subplots(2, 3, figsize=(15, 5))
                fig.canvas.set_window_title(file_path)
                axs = axs.flatten()
            else:
                axs = None
            trajs_repro, diff_tp, diff_ys = evaluate_dmp_contextual(dmp_context, params_and_trajs_all,
                                                                    axs=axs)

            diff_tp = {tp: np.mean(result) for tp, result in diff_tp.items()}
            diff_ys = {tp: np.mean(result) for tp, result in diff_ys.items()}

            diff_tp_all[basename] = diff_tp
            diff_ys_all[basename] = diff_ys

            if save_for_robot:
                for repro in trajs_repro:
                    task_params = repro[0]
                    traj_repro = repro[1]
                    as_matrix = add_orientation(traj_repro)
                    filename = Path("data/coathanger23_output", f"{basename}_context{task_params[0]:.3f}.csv")
                    print(filename)
                    np.savetxt(filename, as_matrix, fmt="%1.5f", delimiter=',')

    return diff_tp_all, diff_ys_all


def run_experiments_cached(main_directory, overwrite=False):
    if Path(main_directory, "diff_tp.json").exists() and not overwrite:
        diff_tp_all = loadjson(Path(main_directory, "diff_tp.json"))
        diff_ys_all = loadjson(Path(main_directory, "diff_ys.json"))
        plot_results_list(diff_tp_all, diff_ys_all)

    else:
        run_experiments(main_directory)

    filename = Path(main_directory, 'results.svg')
    print(f"Saving to {filename}")
    plt.savefig(filename)
    plt.show()


def run_experiments(main_directory):
    experiments = [
        # dmp_type, decoupled
        ("IJSPEERT_2002_MOVEMENT", False),
        # ("IJSPEERT_2002_MOVEMENT", True),
        ("KULVICIUS_2012_JOINING", False),
        # ("KULVICIUS_2012_JOINING", True),
        # ("2022", False),
        ("2022", True),
    ]

    # Parameters of RBFN
    n_batch = 4
    all_n_basis = [10]  # [2, 5, 10]
    for n_basis in all_n_basis:
        diff_tp_all = []
        diff_ys_all = []
        for i_batch in range(n_batch):
            # Load the data
            params_and_trajs = load_data_coathanger(i_batch)

            # Prepare the directory to write the results to
            batch_directory = Path(main_directory, f"batch{i_batch}")
            batch_directory.mkdir(parents=True, exist_ok=True)

            # Run everything for this set of trajectories
            save_for_robot = i_batch == 2 and False
            diff_tp, diff_ys = main_with_trajs(experiments, params_and_trajs, batch_directory,
                                               n_basis=n_basis, save_for_robot=save_for_robot)
            diff_tp_all.append(diff_tp)
            diff_ys_all.append(diff_ys)

            # Comment this in to see the results for each individual batch of demonstrations
            #plot_results(diff_tp, diff_ys)
            #plt.show()

        savejson(Path(main_directory, "diff_tp.json"), diff_tp_all)
        savejson(Path(main_directory, "diff_ys.json"), diff_ys_all)
        plot_results_list(diff_tp_all, diff_ys_all)


def main():
    """ Main function of the script. """
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=300)

    data_type = "coathanger23"
    main_directory = Path(f"results/optimize_contextual_dmp/{data_type}")
    run_experiments_cached(main_directory)


if __name__ == "__main__":
    main()
