import numpy as np
import pandas as pd
import os
import functools
from absl import flags, app

import skopt
from sklearn.linear_model import LinearRegression, Ridge
from skopt.space.space import Real, Integer

from utils import resample_training_data
from utils import compute_nrmse
from utils import expand_output_layer

from delay_ESN import delay_ESN

# define flags
FLAGS = flags.FLAGS

def monitoring(res):
    print("Iteration {}".format(len(res.func_vals)))
    print("Best NRMSE in autonomous continuation = {}".format(np.amin(res.func_vals)))
    ix = np.argmin(res.func_vals)
    print("Best parameter found so far ", res.x_iters[ix])
    print("\n")


def get_test_prediction(desn, clf, input):
    testSteps = 20_000
    input_dim = 2
    prediction = np.zeros((testSteps, input_dim))

    for i in range(testSteps):
        x = desn.eval_esn(input[i])
        state = np.concatenate((x, np.square(x), x**3), axis=0)
        y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
        prediction[i, :] = y_pred

    return prediction


def get_autonomous_continuation(desn, clf, input):
    testSteps = 1_000
    input_dim = 2

    prediction = np.zeros((testSteps, input_dim))
    y_pred = input[0]
    for i in range(testSteps):
        x = desn.eval_esn(y_pred)
        state = np.concatenate((x, np.square(x), x**3), axis=0)
        y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
        prediction[i, :] = y_pred

    return prediction


def eval_esn(params, input=None, target=None, delay=5000, network_size=333, input_dim = 2):
    
    assert input is not None or target is not None, "input and target must be provided"

    desn = delay_ESN(network_size, input_dim, delay)
    desn.set_params(params)

    inputStep = 0
    # vanish initial transients

    initialSteps = 50_000
    for _ in range(initialSteps):
        desn.eval_esn_withNoise(input[inputStep])
        inputStep += 1

    # vanish initial transients
    trainSteps = 60_000
    stateMatrix = []

    for i in range(trainSteps):
        x = desn.eval_esn_withNoise(input[inputStep])
        inputStep += 1
        stateMatrix.append(x)

    trainingTarget = target[initialSteps:initialSteps + trainSteps]

    trainStateMatrix, trainingTarget = resample_training_data(
        stateMatrix, trainingTarget)

    trainStateMatrix = np.concatenate(
        (trainStateMatrix, np.square(trainStateMatrix), trainStateMatrix**3), axis=1)

    clf = Ridge(1e-5)
    clf.fit(trainStateMatrix, trainingTarget)
    error_train = 1 - clf.score(trainStateMatrix, trainingTarget)
    print(f"Training Error (NRMSE)={error_train}")

    prediction = get_autonomous_continuation(desn, clf, input[inputStep:])
    testingTarget = target[inputStep:]

    return compute_nrmse(testingTarget, prediction)

flags.DEFINE_integer("delay", 5000, "delay")
flags.DEFINE_integer("network_size", 333, "network size")
flags.DEFINE_integer("input_dim", 2, "input_dim")
flags.DEFINE_integer("workers", 8, "workers")


def main(_):
    
    df = pd.read_csv("./data/amplitude.txt",
                 header=None, skiprows=2, nrows=1_000_000)
    input = np.array(df.iloc[100_000:1_000_000, :2])
    input = input[:, :]
    input -= np.mean(input)
    input /= np.std(input)
    target = np.array(input[1:, :])

    borderParam = []
    borderParam.append(Real(0.01, 1, prior="uniform"))
    borderParam.append(Real(0.1, 1, prior="uniform"))
    borderParam.append(Real(0.1, 1, prior="uniform"))
    borderParam.append(Real(0.01, 10, prior="log-uniform"))
    
    _eval_esn = functools.partial(eval_esn, 
                                  input=input,
                                  target=target,
                                  delay=FLAGS.delay,
                                  network_size=FLAGS.network_size,
                                  input_dim =FLAGS.input_dim)
    

    results = skopt.gp_minimize(_eval_esn,
                                borderParam,
                                base_estimator="GP",
                                acq_func="EI",
                                # the acquisition function
                                n_calls=100,         # the number of evaluations of f
                                n_random_starts=20,  # the number of random initialization points
                                noise=1e-3,
                                n_jobs=FLAGS.workers,
                                acq_optimizer='lbfgs',
                                n_restarts_optimizer=100,
                                callback=[monitoring]
                                )

    del results.specs['args']['func']
    del results.specs['args']['callback']
    del results.models
    del results.random_state
    skopt.dump(results, "testBayes.csv")
    print(results.x)

if __name__ == '__main__':
    app.run(main)