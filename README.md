Robot Base Pose Optimization
============================

This repository contains all the code required to run and analyze the experiments of the paper:

"Decreasing Robotic Cycle Time via Base-Pose Optimization", by Matthias Mayer and Matthias Althoff, 2024.

Feel free to send an [E-Mail](mailto:matthias.mayer@tum.de?subject=BaseOptPaper) to receive a preprint of the paper.

Install
=======

This repository uses [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) and [git lfs](https://git-lfs.com/).
It is easiest if you install git lfs first, e.g., on Ubuntu `(sudo) apt-get install git-lfs`.
Then clone the repository with `git clone --recurse-submodules <repo link>`.
Check if the csv files in the [data](data/outdir_test_1200/outdir_hard_1200/Space/xyz/Set/test_hard/Alg/BOOptimizer/Seed/1_raw.csv) folder are not empty (otherwise lfs failed) and that [mcs](mcs) is non-empty (otherwise git submodules failed).

0. We suggest setting up a conda environment for better control and isolation: `conda create --name rbpo python==3.11`.
Activate the environment with `conda activate rbpo`.
1. Then install the requirements: `pip install -r requirements.txt`.
We install this package and [mcs](https://gitlab.lrz.de/tum-cps/mcs) (providing utilities such as path planning) in editable mode, so that changes to the code are immediately reflected in the installed package.

Running the Experiments
=======================

The experiments are best run inside the provided docker container.
Use the provided docker-compose file to get a shell with all dependencies installed: `docker compose run -i --service-ports base-opt bash`.
Then you can run the experiments with the provided shell scripts in the `experiments` folder.
The experiments have two parts:

1. Hyperparameter optimization [run_hyperparameter_optimization.sh](experiments/run_hyperparameter_optimization.sh).
The optuna database for our training is given in [data/optuna.db](data/optuna.db).
It can be viewed, e.g., with the optuna dashboard: `optuna-dashboard --port 6006 sqlite:///data/optuna.db`.
2. Evaluation of the best hyperparameters [run_evaluation.sh](experiments/run_evaluation.sh)
We suggest running the evaluation with [GNU parallel](https://www.gnu.org/software/parallel/) over task sets/action spaces/seed/algorithm combinations as done in [run_evaluation.sh](experiments/run_evaluation.sh).
The main output are a set of csv files per task set, action space, algorithm, and seed combination, logs of errors/stdout per run are stored alongside.
The results for our analysis are provided in the [data/outdir_test_1200 folder](data/outdir_test_1200); our resulting data is provided as a [zip](data/outdir_test_1200.zip).
The best solutions uploaded to [CoBRA](https://cobra.cps.cit.tum.de) are provided in [best_solutions.zip](data/best_solutions.zip).

We note that the experiments are computationally expensive and require a lot of CPU time and memory (order of 1000s CPU hours).

Running the Evaluation
======================

The evaluations for the paper are provided as a [jupyter notebook](evaluation/evaluate_all.ipynb).
These can be run just by installing the requirements, starting jupyter lab `jupyter lab` and running the notebook.
The notebook uses the csv files in [data/outdir_test_1200 folder](data/outdir_test_1200) to create all plots and tables used in the paper.

Issue Tracker and Questions
===========================

Please use the [issue tracker of this repository](https://gitlab.lrz.de/tum-cps/robot-base-pose-optimization/-/issues) to report bugs or ask questions.

Unittests and CI
================

We use pytest for unittests and gitlab CI for continuous integration.
The used docker image can be locally built with `ci/build_docker.sh`.
To run the unittests via PyCharm make sure you set the working directory to the root of the repository and we suggest using the docker compose interpreter to easily test the OMPL integrations.

Contact
=======

Matthias Mayer [matthias.mayer@tum.de](mailto:matthias.mayer@tum.de)
