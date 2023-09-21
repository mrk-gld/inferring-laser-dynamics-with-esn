#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:14:10 2022

@author: mirko
"""
import numpy as np


def compute_nrmse(target, prediction):
    num_samples = len(prediction)
    error = 0.0
    for y_hat, y in zip(target, prediction):
        error += np.mean(y_hat - y)**2
    error /= num_samples
    error = np.sqrt(error)
    return error


def expand_output_layer(states, max_order=2, axis=0):
    states = np.array(states)
    x = np.copy(states)
    for i in range(2, max_order+1):
        x = np.concatenate((x, states**i), axis=axis)
    return x

def resample_training_data(stateMatrix, trainingTarget, sampling_factor=5):

    stateMatrix = np.array(stateMatrix)
    trainSteps = stateMatrix.shape[0]
    idx = np.arange(trainSteps)
    np.random.shuffle(idx)
    idx = idx[::sampling_factor]

    trainStateMatrix = stateMatrix[idx, :]

    trainingTarget = trainingTarget[idx, :]
    return trainStateMatrix, trainingTarget