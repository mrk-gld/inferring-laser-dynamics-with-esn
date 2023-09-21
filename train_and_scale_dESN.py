import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange
from absl import flags, app
from sklearn.linear_model import LinearRegression

from delay_ESN import delay_ESN

from utils import compute_nrmse
from utils import expand_output_layer



FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "data/amplitude.txt", "path to data file")

flags.DEFINE_integer("initial_steps", 50_000, "number of initial steps to vanish transients")
flags.DEFINE_integer("train_steps", 100_000, "number of training steps")
flags.DEFINE_integer("test_steps", 20_000, "number of test steps")
flags.DEFINE_integer("autonomous_steps", 60_000, "number of autonomous continuation steps")
flags.DEFINE_integer("saving_steps", 100_000, "number of steps to save attractor images")

flags.DEFINE_integer("delay", 5_000, "delay")
flags.DEFINE_integer("network_size", 333, "network size")
flags.DEFINE_integer("input_dim", 2, "input dimension")
flags.DEFINE_integer("output_exp", 4, "output expansion")

flags.DEFINE_float("training_noise", 1e-6, "training noise")
                    

def main(_):
    
    os.makedirs("attractor_images", exist_ok=True)
    os.makedirs("attractor_data", exist_ok=True)
    
    df = pd.read_csv(FLAGS.data_path, 
                    header=None, 
                    skiprows=2,
                    nrows=1_800_000)
    
    input = np.array(df.iloc[100_000:1_800_000, :2])
    input = input[:, :]
    input -= np.mean(input)
    input /= np.std(input)
    target = np.array(input[1:, :])

    params = [0.11,  0.58025,  0.00012663801734674035, 0.9238461538461539]

    desn = delay_ESN(FLAGS.network_size, FLAGS.input_dim, FLAGS.delay)
    desn.set_params(params)
    desn.training_noise = FLAGS.training_noise

    inputStep = 0
    # vanish initial transients
    for i in range(FLAGS.initial_steps):
        desn.eval_esn_withNoise(input[inputStep])
        inputStep += 1

    # collect state matrix
    stateMatrix = []
    for i in trange(FLAGS.train_steps, desc="Collecting state matrix"):
        x = desn.eval_esn_withNoise(input[inputStep])
        inputStep += 1
        stateMatrix.append(x)

    idx = np.arange(FLAGS.train_steps)
    np.random.shuffle(idx)
    idx = idx[:]

    stateMatrix = np.array(stateMatrix)

    stateMatrix = stateMatrix[idx, :]
    trainStateMatrix = stateMatrix
    trainStateMatrix = expand_output_layer(
        trainStateMatrix, max_order=FLAGS.output_exp, axis=1)

    trainingTarget = target[FLAGS.initial_steps:FLAGS.initial_steps + FLAGS.train_steps]
    trainingTarget = trainingTarget[idx, :]

    clf = LinearRegression()
    clf.fit(trainStateMatrix, trainingTarget)
    error_train = 1 - clf.score(trainStateMatrix, trainingTarget)
    print(f"Training error (NRMSE) = {error_train}")

    FLAGS.test_steps = 20_000
    targetAuto = np.zeros((FLAGS.test_steps, FLAGS.input_dim))
    prediction = np.zeros((FLAGS.test_steps, FLAGS.input_dim))

    for i in trange(FLAGS.test_steps,desc="Testing"):
        x = desn.eval_esn(input[inputStep])
        state = expand_output_layer(x, FLAGS.output_exp)
        y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
        targetAuto[i, :] = target[inputStep]
        prediction[i, :] = y_pred
        inputStep += 1

    print("Testing error (NRMSE) = ", compute_nrmse(targetAuto, prediction))

    y_pred = clf.predict(np.reshape(
        state, newshape=(1, -1)))[0]

    targetAuto = np.zeros((FLAGS.autonomous_steps, FLAGS.input_dim))
    prediction = np.zeros((FLAGS.autonomous_steps, FLAGS.input_dim))

    for i in trange(FLAGS.autonomous_steps, desc="Autonomous continuation"):
        x = desn.eval_esn(y_pred)
        state = expand_output_layer(x, FLAGS.output_exp)
        y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
        targetAuto[i, :] = target[inputStep]
        prediction[i, :] = y_pred
        inputStep += 1

    corr_real = np.corrcoef(targetAuto[:, 0], prediction[:, 0])
    corr_imag = np.corrcoef(targetAuto[:, 1], prediction[:, 1])
    print("Abs. Lin. Correlation real part E(t) = ", abs(corr_real[0, 1]))
    print("Abs. Lin. Correlation imag. part E(t) = ", abs(corr_imag[0, 1]))

    plt.figure(figsize=(6, 3))
    plot_length = 6_000
    t = np.linspace(0, plot_length*2e-12, plot_length)
    plt.plot(t, targetAuto[:plot_length, 0], c="blue", ls="solid", label=r"E_r")
    plt.plot(t, prediction[:plot_length, 0],
            c="darkblue", ls="dashed", label=r"E_r dESN")

    plt.plot(t, targetAuto[:plot_length, 1], c="red", ls="solid", label=r"E_i")
    plt.plot(t, prediction[:plot_length, 1],
            c="darkred", ls="dashed", label=r"E_i dESN")

    plt.legend()
    plt.xlabel("time in ns")
    plt.ylabel("norm. elec. field comp.")
    plt.tight_layout()
    # plt.savefig("field_components.png",dpi=333, bbox_inches="tight")

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(121)
    amp = np.sqrt(targetAuto[:, 0]**2 + targetAuto[:, 1]**2)
    plt.plot(amp[FLAGS.delay:], amp[:-FLAGS.delay], c="darkblue", label="original", lw=0.2)
    plt.ylabel(r"$x(t-\tau)$")
    plt.xlabel(r"$x(t)$")

    ax = fig.add_subplot(122)
    amp = np.sqrt(prediction[:, 0]**2 + prediction[:, 1]**2)
    plt.plot(amp[FLAGS.delay:], amp[:-FLAGS.delay], c="darkgreen", label="auto. pred.", lw=0.2)
    plt.ylabel(r"$x(t-\tau)$")
    plt.xlabel(r"$x(t)$")

    plt.legend()
    plt.tight_layout()
    # plt.savefig("attractor_comparison.png",dpi=333, bbox_inches="tight")

    plt.figure(figsize=(6, 3))
    amp1 = np.sqrt(targetAuto[:, 0]**2 + targetAuto[:, 1]**2)
    t = np.linspace(0, len(targetAuto)*2e-12, len(targetAuto))
    plt.plot(t, amp1, c="darkblue")

    amp2 = np.sqrt(prediction[:, 0]**2 + prediction[:, 1]**2)
    t = np.linspace(0, len(prediction)*2e-12, len(prediction))
    plt.plot(t, amp2, c="darkgreen", ls="dashed")

    plt.ylabel("|E(t)|")
    plt.xlabel("time in s")
    plt.tight_layout()
    # plt.savefig("amplitudes_t.png",dpi=333, bbox_inches="tight")
    plt.show()

    scan_lengths = [400, 25, 25, 25]
    delay = FLAGS.delay
    for i in trange(4,desc="Scanning delay",leave=True):

        for q in trange(scan_lengths[i],
                        desc="eval esn while shortening delay,
                        leave=True):
            
            delay = delay - 10
            desn.reset_delay(delay)

            for k in range(int(0.5 * desn.n_delay)):
                x = desn.eval_esn(y_pred)
                state = expand_output_layer(x, FLAGS.output_exp)
                y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]

        stabilize_steps = 15*delay
        prediction = np.empty((stabilize_steps, FLAGS.input_dim))
        for q in range(stabilize_steps):
            x = desn.eval_esn(y_pred)
            state = expand_output_layer(x, FLAGS.output_exp)
            y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
            prediction[q, :] = y_pred

        prediction = np.zeros((FLAGS.saving_steps, FLAGS.input_dim))

        for q in trange(FLAGS.saving_steps,desc="eval esn for plots at delay {}".format(delay)):
            x = desn.eval_esn(y_pred)
            state = expand_output_layer(x, FLAGS.output_exp)
            y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
            prediction[q, :] = y_pred

        fig = plt.figure(figsize=(4, 4))
        ax2 = fig.add_subplot(111)
        predictionPlot = prediction[:FLAGS.autonomous_steps, :]
        amp = np.sqrt(predictionPlot[:, 0]**2 + predictionPlot[:, 1]**2)
        ax2.plot(amp[delay:], amp[:-delay], c="darkgreen",
                label="auto. pred.", lw=0.2)
        ax2.set_title(r" delay $\tau=$={:.2e} s".format(delay*2e-12))
        ax2.set_ylabel(r"$I(t-\tau)$")
        ax2.set_xlabel(r"$I(t)$")


        ax2.legend()
        plt.tight_layout()
        figname = "attractor_images/attractor_{}_ns.png".format(delay*2)
        plt.savefig(figname, dpi=333, bbox_inches="tight")
        np.save("attractor_data/attractor_{}_ns.npy".format(delay*2), prediction)
        plt.close()


if __name__ == "__main__":
    app.run(main)