""" Module to show differences between DMP formulations."""
import pprint

from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem
from dmpbbo.dynamicalsystems.SigmoidSystem import SigmoidSystem
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem
from dmpbbo.dynamicalsystems.TimeSystem import TimeSystem

from save_plot import save_plot
from utils import *


def get_dmp_local(demonstration, dmp_type, fa_name, n_basis=30):
    """
    Get a DMP (version local to this module)

    @param demonstration: the demonstrated trajectory
    @param dmp_type: The DMP formulation
    @param fa_name: The name of the function approximator.
    @param n_basis: The number of basis functions for the function approximator.
    @return:
    """
    # Prepare the function approximators
    n_dofs = demonstration.dim
    function_apps = get_function_apps(n_dofs, fa_name, n_basis)

    # Prepare the dynamical systems
    y_init = demonstration.y_init
    y_attr = demonstration.y_final
    tau = demonstration.duration
    dmp_args_kulvicius = {
        "goal_system": ExponentialSystem(tau, y_init, y_attr, 15),
        "phase_system": TimeSystem(tau, True),
        "gating_system": SigmoidSystem(tau, 1, -20.0, 0.9),
    }

    if "KULVICIUS" in dmp_type:
        dmp_args = dmp_args_kulvicius

    elif "IJSPEERT" in dmp_type:
        dmp_args = {
            "gating_system": ExponentialSystem(tau, 1, 0, 4),
            "phase_system": ExponentialSystem(tau, 1, 0, 4),
            "goal_system": None,
        }

        if "PHASE" in dmp_type:
            dmp_args["phase_system"] = dmp_args_kulvicius["phase_system"]
        if "GOAL" in dmp_type:
            dmp_args["goal_system"] = dmp_args_kulvicius["goal_system"]
        if "GATING" in dmp_type:
            dmp_args["gating_system"] = dmp_args_kulvicius["gating_system"]

    else:
        raise Exception("Unknown dmp_type: " + dmp_type)

    if dmp_args["goal_system"]:
        dmp_args["goal_system"].alpha = 10
    spring_constant = 35
    damping_coefficient = 2 * np.sqrt(spring_constant)
    mass = 1.0
    dmp_args["transformation_system"] = SpringDamperSystem(
        tau, y_init, y_attr, damping_coefficient, spring_constant, mass
    )

    # Train the DMP
    dmp_args["save_training_data"] = True
    dmp = Dmp.from_traj(demonstration, function_apps, **dmp_args)

    return dmp


def main():
    """Run one demo for bbo_of_dmps.
    """

    # demo_name = "stulp09compact"
    demo_name = "stulp13learning_meka"
    traj_number = 4
    traj_demo = get_demonstration(demo_name, traj_number=traj_number)

    dmp_types = [
        ["IJSPEERT"],
        ["IJSPEERT", "PHASE"],
        ["IJSPEERT", "PHASE", "GATING"],
        ["KULVICIUS"],
    ]
    fa_names = [
        "RBFN",
        #        "LWR"
    ]

    n_plots = 6
    n_rows = len(dmp_types)
    n_cols = n_plots
    ratio = 1.61

    mae_all = {}
    for n_basis in [30]:
        for fa_name in fa_names:
            print(f"{fa_name} {n_basis}")
            i_row = 0
            fig = plt.figure(figsize=(ratio * n_cols * 3, n_rows * 3))

            t = f"Analysis of parameters ({fa_name} {n_basis})"
            fig.canvas.set_window_title(t)

            ylims_targets = None
            ylims_params = None
            for dmp_type in dmp_types:
                print(dmp_type)

                dmp = get_dmp_local(traj_demo, dmp_type, fa_name, n_basis)
                if fa_name == "RBFN":
                    dmp.set_selected_param_names("weights")
                else:
                    dmp.set_selected_param_names(["offsets", "slopes"])

                ts = np.linspace(0.0, 1.25 * dmp.tau, 151)

                plot_dmp_integration = False
                if plot_dmp_integration:
                    dmp.plot(ts)
                    t = f"Comparison between demonstration and reproduced ({dmp_type}, {fa_name} {n_basis})"
                    plt.gcf().canvas.set_window_title(t)

                axs = [
                    fig.add_subplot(n_rows, n_plots, i_row * n_plots + j + 1)
                    for j in range(n_plots)
                ]

                plot_dmp_parameterization(traj_demo, ts, dmp, axs, True)

                axs[0].text(0.1 * dmp.tau, 0.5, "-".join(dmp_type))
                if ylims_targets is None:
                    ylims_targets = axs[3].get_ylim()
                axs[3].set_ylim(ylims_targets)
                if ylims_params is None:
                    ylims_params = axs[4].get_ylim()
                axs[4].set_ylim(ylims_params)
                axs[5].set_ylim(ylims_params)

                mae = mae_demonstration_reproduced(traj_demo, dmp)
                mae_all["-".join(dmp_type) + "_" + fa_name + f"_{n_basis}"] = mae
                i_row += 1

    pprint.pprint(mae_all)
    save_plot(__file__)
    plt.show()


if __name__ == "__main__":
    main()
