Robot Base Pose Optimization
============================

This repository contains all codes required to run and analyze the experiments of the paper:

"Decreasing Robotic Cycle Time via Base-Pose Optimization", by Matthias Mayer and Matthias Althoff

Install
=======

0. We suggest setting up a conda environment for better control and isolation: `conda create --name rbpo python==3.11`.
1. Then install the requirements: `pip install -r requirements.txt`. We install this package and mcs in editable mode, so that changes to the code are immediately reflected in the installed package.

Running the Experiments
=======================

The experiments are best run inside a docker container. Use the provided docker-compose file to get a shell with all dependencies installed: `docker compose run -i --service-ports base-opt bash`.
Then you can run the experiments with the provided shell scripts in the `experiments` folder.
The experiments have two parts:

1. Hyperparameter optimization [run_hyperparameter_optimization.sh](experiments/run_hyperparameter_optimization.sh).
The optuna database for our training is given in [data/optuna.db](data/optuna.db).
It can be viewed, e.g., with the optuna dashboard: `optuna-dashboard --port 6006 sqlite:///data/optuna.db`.
2. Evaluation of the best hyperparameters [run_evaluation.sh](experiments/run_evaluation.sh)

We note that the experiments are computationally expensive and require a lot of CPU time and memory (order of 1000s CPU hours).
We suggest running the evaluation with [GNU parallel](https://www.gnu.org/software/parallel/) over task sets/action spaces/seed/algorithm combinations as done in [run_evaluation.sh](experiments/run_evaluation.sh).
The results for the analysis are provided in the [data/outdir_test_1200 folder](data).

Running the Evaluation
======================

The evaluations for the paper are provided as a [jupyter notebook](tba).
These can be run just by installing the requirements, starting jupyter lab `jupyter lab` and running the notebook.

Issue Tracker and Questions
===========================

Please use the [issue tracker of this repository](https://gitlab.lrz.de/tum-cps/robot-base-pose-optimization/-/issues) to report bugs or ask questions.

Unittests and CI
================

We use pytest for unittests and gitlab CI for continuous integration.
The used docker image can be locally built with `ci/build_docker.sh`.
To run the unittests via PyCharm make sure you set the working directory to the root of the repository.

Contact
=======
Matthias Mayer (matthias.mayer@tum.de)
