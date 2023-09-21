import numpy as np
import pandas as pd
from scipy.sparse import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import skopt
from skopt.space.space import Real, Integer
from tqdm import trange
from delay_ESN import *
import matplotlib.gridspec as gridspec
from utils import *
import matplotlib

df = pd.read_csv("../../data/E/amplitude.txt",
                 header=None, skiprows=2, nrows=1_000_000)
input = np.array(df.iloc[100_000:1_000_000, :2])
input = input[:, :]
input -= np.mean(input)
input /= np.std(input)
target = np.array(input[1:, :])

# B
# params = [0.8807546708786556, 0.9198590362836981, 0.5146926370516434, 0.013590834031695156]
params = [0.8542912758754485, 1.25, 0.06, 0.03]
params = [0.11,  0.58025,  0.00012663801734674035,
          0.9238461538461539]

delay = 5000
networkSize = 333
input_dim = 2

desn = delay_ESN(networkSize, input_dim, delay)
desn.set_params(params)
desn.training_noise = 1e-5

inputStep = 0
# vanish initial transients

initialSteps = 16_000
for i in range(initialSteps):
    desn.eval_esn_withNoise(input[inputStep])
    inputStep += 1

# vanish initial transients
trainSteps = 125_000
sampling_steps = 2
stateMatrix = []
for i in trange(trainSteps):
    x = desn.eval_esn_withNoise(input[inputStep])
    inputStep += 1
    stateMatrix.append(x)

idx = np.arange(trainSteps)
np.random.shuffle(idx)
idx = idx[::sampling_steps]

stateMatrix = np.array(stateMatrix)
lastRow_stateMatrix = stateMatrix[-1, :]
lastRow_stateMatrix = np.concatenate((lastRow_stateMatrix, np.square(
    lastRow_stateMatrix), lastRow_stateMatrix**3), axis=0)
trainStateMatrix = stateMatrix  # [idx, :]
trainStateMatrix = np.concatenate((trainStateMatrix, np.square(
    trainStateMatrix), trainStateMatrix**3, trainStateMatrix**4), axis=1)

trainingTarget = target[initialSteps:initialSteps + trainSteps]
trainingTarget = trainingTarget  # [idx, :]

clf = LinearRegression()
clf.fit(trainStateMatrix, trainingTarget)
error_train = 1 - clf.score(trainStateMatrix, trainingTarget)
print(f"error={error_train}")

testSteps = 20_000
targetAuto = np.zeros((testSteps, input_dim))
prediction = np.zeros((testSteps, input_dim))

for i in trange(testSteps):
    x = desn.eval_esn(input[inputStep])
    state = np.concatenate((x, np.square(x), x**3, x**4), axis=0)
    y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
    targetAuto[i, :] = target[inputStep]
    prediction[i, :] = y_pred
    inputStep += 1

print(compute_nrmse(targetAuto, prediction))

y_pred = clf.predict(np.reshape(
    state, newshape=(1, -1)))[0]

autonomousSteps = 100_000
targetAuto = np.zeros((autonomousSteps, input_dim))
prediction = np.zeros((autonomousSteps, input_dim))

for i in trange(autonomousSteps):
    x = desn.eval_esn(y_pred)
    state = np.concatenate((x, np.square(x), x**3, x**4), axis=0)
    y_pred = clf.predict(np.reshape(state, newshape=(1, -1)))[0]
    targetAuto[i, :] = target[inputStep]
    prediction[i, :] = y_pred
    inputStep += 1

corr_real = np.corrcoef(targetAuto[:, 0], prediction[:, 0])
corr_imag = np.corrcoef(targetAuto[:, 1], prediction[:, 1])
print(abs(corr_real[0, 1]))
print(abs(corr_imag[0, 1]))

# %% plot electrical field
font = {'size': 10}

matplotlib.rc('font', **font)
fig = plt.figure(figsize=(4, 7))

spec2 = gridspec.GridSpec(ncols=2, nrows=3, figure=fig,
                          height_ratios=[1, 1, 1.5], hspace=0.4, wspace=0.05)
f2_ax1 = fig.add_subplot(spec2[0, :])
f2_ax1.set_title("a)", loc="left")
# f2_ax2 = fig.add_subplot(spec2[0, 2])
f2_ax4 = fig.add_subplot(spec2[1, :])
f2_ax4.set_title("b)", loc="left")

f2_ax2 = fig.add_subplot(spec2[2, 0])
f2_ax2.set_title("c)", loc="left")
f2_ax3 = fig.add_subplot(spec2[2, 1])
f2_ax3.set_title("d)", loc="left")
# f2_ax5 = fig.add_subplot(spec2[1, 2:])

plot_length = 6_000
t = np.linspace(0, plot_length*0.002, plot_length)
f2_ax1.plot(t, targetAuto[:plot_length, 0],
            c="blue", ls="solid", label=r"$E_r^{LK}$")
f2_ax1.plot(t, prediction[:plot_length, 0],
            c="darkblue", ls="dashed", label=r"$E_r^{dESN}$")

f2_ax1.plot(t, targetAuto[:plot_length, 1],
            c="red", ls="solid", label=r"$E_i^{LK}$")
f2_ax1.plot(t, prediction[:plot_length, 1],
            c="darkred", ls="dashed", label=r"$E_i^{dESN}$")

f2_ax1.set_xlabel("t in ns")
f2_ax1.set_ylabel(r"norm. $E_{r,i}(t)$,$\hat{E}_{r,i}(k)$")
f2_ax1.set_ylim(-4, 2)
f2_ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
f2_ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
f2_ax1.legend(frameon=False, ncols=4, loc="lower left",
              handletextpad=0.2, columnspacing=1)
f2_ax1.set_xlim(0, 12.5)
# plt.tight_layout()
# plt.savefig("field_components.png", dpi=333, bbox_inches="tight")

# plot attractor in delay embedding
dot_size = 0.02
alpha = 0.1
amp = np.sqrt(targetAuto[:, 0]**2 + targetAuto[:, 1]**2)
f2_ax2.scatter(amp[delay:], amp[:-delay],
               c="darkblue", s=dot_size, alpha=alpha)

f2_ax2.set_ylabel(r"$|E(t-\tau)|$")
f2_ax2.set_xlabel(r"$|E(t)|$")
f2_ax2.set_xlim(0.75, 2.25)
f2_ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
f2_ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
f2_ax2.set_ylim(0.75, 2.25)
f2_ax2.set_aspect('equal', 'box')

amp = np.sqrt(prediction[:, 0]**2 + prediction[:, 1]**2)
# f2_ax3.plot(amp[delay:], amp[:-delay], c="darkgreen",
# label="auto. pred.", lw=0.2)
f2_ax3.scatter(amp[delay:], amp[:-delay], c="darkgreen",
               s=dot_size, alpha=alpha)
f2_ax3.set_aspect('equal', 'box')
f2_ax3.set_xlabel(r"$|E(t)|$")
f2_ax3.set_yticklabels([])
f2_ax3.set_xlim(0.75, 2.25)
f2_ax3.set_ylim(0.75, 2.25)
f2_ax3.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
f2_ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

plt_length = 40_000
amp1 = np.sqrt(targetAuto[:plt_length, 0]**2 + targetAuto[:plt_length, 1]**2)
t = np.linspace(0, (plt_length)*0.002, (plt_length))
f2_ax4.plot(t, amp1, c="darkblue", label="original", lw=1)

amp2 = np.sqrt(prediction[:plt_length, 0]**2 + prediction[:plt_length, 1]**2)
t = np.linspace(0, (plt_length)*0.002, (plt_length))
f2_ax4.plot(t, amp2, c="darkgreen", ls="dashed", label="pred.", lw=1)

f2_ax4.set_ylabel(r"$|E(t)|$, $|\hat{E}(k)|$")
f2_ax4.set_xlabel("t in ns")
f2_ax4.set_ylim(0.5, 2.25)
f2_ax4.set_xlim(0, 80)
f2_ax4.xaxis.set_minor_locator(plt.MultipleLocator(5))
f2_ax4.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
f2_ax4.legend(frameon=False, ncols=2)

fig.savefig("testing_prediction_overview.png", dpi=333, bbox_inches="tight")
plt.show()


# %%
# plot optical spectra
plt.figure()
colors = ["darkblue", "darkgreen"]
ls = ["dotted", "solid"]
for i, field in enumerate([targetAuto, prediction]):
    field = field[:, 0] + 1j * field[:, 1]
    freq = np.fft.fftfreq(len(field), d=2e-12)
    fft = np.fft.fft(field)
    plt.semilogy(freq, abs(fft), c=colors[i], ls=ls[i], lw=0.5, alpha=0.8)
    # plt.savefig(f"{folder}/optical_spectra.png", dpi=333, bbox_inches="tight")

plt.xlim(-0.06e11, 0.06e11)
plt.ylim(1, 10**5)


# %%
