# dmpbbo_sct_experiments

Repo for the experiments in the paper in which a novel DMP formulation is proposed. 

Running this code requires v2.1 of dmpbbo to be installed: https://github.com/stulp/dmpbbo/releases/tag/v2.1.0

The main functionality for the novel formulation became available in V2.1.0 of dmpbbo, see for instance:
* https://github.com/stulp/dmpbbo/blob/c73e1418b611f31cfbe652df51f14f8c18552897/dmpbbo/dmps/Dmp.py#L101

## Contents of the repo

This repo provides experiments which apply this novel formulation to different datasets, which can be found in the data/ directory.

The following Python scripts are provided:

### Demos and illustrations

* `illustrate_from_ijspeert_to_kulvicius.py`: Show the effects of different DMP formulations on the distribution of function approximator parameters.
* `illustrate_richards.py`: Illustrate the  generalized logisitics function, also known as _Richard's function_.
* `illustrate_novel_dyn_systems.py`: Illustrate the novel formulation with the generalized logisitics function. 
* `demo_optimize_dyn_sys_parameters.py`: A generic script for optimizing the parameters of the dynamical systems in a DMP.
* `presentation_training.py` and `presentation_optimization.py`: Generate plots for the ICRA presentation.
 
### Experiments

* `experiment_bbo_of_dyn_systems.py`: Experiment 1 from the paper, i.e. train the different formulations on different datasets. 
* `experiment_optimize_contextual_dmp.py`: Experiment 2 from the paper, i.e. train a contextual DMP with the different underlying formulations on the coathanger dataset. 
* `experiment_from_ijspeert_to_kulvicius.py`: Bonus experiment: visualize more extensively the impact of the different formulations.

### Helper modules

* `get_params_and_trajs.py`: Generate synthetic data (not used in the experiments)  
* `load_data_coathanger.py`: Load the coathanger data.
* `save_plot.py`: Module with convenience function for saving a plot to SVG or other formats. 
* `colorpallette.py`: Some default colors.
* `utils.py`: Various utility functions.

