# Inferring Dynamics of a Laser with delayed feedback

- the results of this project are presented on the NOLTA 2023 conference

## Abstract

Lasers with delayed self-feedback can exhibit complex dynamical behavior up to high-dimensional chaos. We exploit the reservoir computing paradigm to train a recurrent neural network to predict the dynamical behavior of a laser with delayed feedback subject to short and long delays. To better adapt the system to the task's characteristics, we incorporate a delay into the recurrent network. After training, we show that the reservoir is capable of predicting both the near-future dynamics and the long-term attractor of the chaotic laser system. Furthermore, by exploiting the symmetry in the laser system, we can change the reservoir's topology while running in the autonomous mode to infer dynamics related to other delay lengths not covered in the training data set. Using this method, we build a single digital twin that can emulate the high dimensional complex dynamics of lasers for different delay lengths and predict beyond the dynamical regime obtained during the training phase while relying on limited data.

## Usage of the repository

- optimization of the reservoir parameters: `bayesian_optimization_dESN.py`
- training and scaling/inference of the dESN: `train_and_scale_dESN.py` 