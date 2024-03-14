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

The experiments are best run inside a docker container. Use the provided docker-compose file to get a shell with all dependencies installed: `docker compose run -i --service-ports rbpo bash`.
Then you can run the experiments with the provided shell scripts in the `experiments` folder.
We note that the experiments are computationally expensive and require a lot of CPU time and memory (order of 1000 CPU hours).
The results for the analysis are provided in the [data folder](data).

Running the Evaluation
======================

The evaluations for the paper are provided as a [jupyter notebook](tba).

Issue Tracker and Questions
===========================

Please use the [issue tracker of this repository](https://gitlab.lrz.de/tum-cps/robot-base-pose-optimization/-/issues) to report bugs or ask questions.

Unittests and CI
================

We use pytest for unittests and gitlab CI for continuous integration.

Contact
=======
Matthias Mayer (matthias.mayer@tum.de)
