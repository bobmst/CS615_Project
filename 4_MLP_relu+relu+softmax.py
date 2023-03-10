# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly

import test
import mylayers
import util


# %% [markdown]
# ## Read data

# %%
df_train = pd.read_csv("./data/mnist_train.csv", header=None)
df_valid = pd.read_csv("./data/mnist_test.csv", header=None)


# %% [markdown]
# ### Shuffle the data set

# %%
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)


# %% [markdown]
# ### Seperate feature column and target columns

# %%
X_train = np.array(df_train.drop(0, axis=1), dtype="f")
Y_train = df_train.loc[:, 0]
X_valid = np.array(df_valid.drop(0, axis=1), dtype="f")
Y_valid = df_valid.loc[:, 0]


# %% [markdown]
# ### One hot encoding Y

# %%
encoder = util.onehot_encoder(Y_train)
Y_train = encoder.encoding(Y_train)
Y_valid = encoder.encoding(Y_valid)


# %%
# Testing
# X_train = X_train[:1000]
# Y_train = Y_train[:1000]


# %% [markdown]
# ## Pretrain

# %% [markdown]
# ### First FC

# %%
X_train_z_scored = mylayers.InputLayer(X_train).forward(X_train)
layers_pre_1 = []
layers_pre_1.append(mylayers.InputLayer(X_train))
# ---In---
fc_layer = mylayers.FullyConnectedLayer(X_train.shape[1], 128)
layers_pre_1.append(fc_layer)
# ---Out---
fc_layer = mylayers.FullyConnectedLayer(128, X_train.shape[1])
layers_pre_1.append(fc_layer)
# ---valid---
layers_pre_1.append(mylayers.SquaredError())

# %%
hyperparams = {
    "epoch": 100,
    "eta": 10e-4,
    "batch_size": 1000,
    "rho1": 0.9,
    "rho2": 0.999,
}
epoch = hyperparams["epoch"]
eta = hyperparams["eta"]
batch_size = hyperparams["batch_size"]
rho1 = hyperparams["rho1"]
rho2 = hyperparams["rho2"]
ls_scores_pre_1, Y_h, _ = test.adam_learn(
    X_train,
    X_train_z_scored,
    layers_pre_1,
    mylayers.SquaredError,
    epoch=epoch,
    eta=eta,
    print_eval=False,
    print_layer=False,
    batch_size=batch_size,
)


# %%
util.draw_scores(ls_scores_pre_1, "None", hyperparams, "SE Score")

# %%
h_1 = layers_pre_1[1].getPrevOut()

# %% [markdown]
# ### Second FC

# %%
layers_pre_2 = []
layers_pre_2.append(mylayers.InputLayer(h_1, z_score=False))
# ---In---
fc_layer = mylayers.FullyConnectedLayer(128, 64)
layers_pre_2.append(fc_layer)
# ---Out---
fc_layer = mylayers.FullyConnectedLayer(64, 128)
layers_pre_2.append(fc_layer)
# ---valid---
layers_pre_2.append(mylayers.SquaredError())

# %%
hyperparams = {
    "epoch": 1000,
    "eta": 10e-5,
    "batch_size": 1000,
    "rho1": 0.9,
    "rho2": 0.999,
}
epoch = hyperparams["epoch"]
eta = hyperparams["eta"]
batch_size = hyperparams["batch_size"]
rho1 = hyperparams["rho1"]
rho2 = hyperparams["rho2"]
ls_scores_train, Y_hat, acc = test.adam_learn(
    h_1,
    h_1,
    layers_pre_2,
    mylayers.SquaredError,
    epoch=epoch,
    eta=eta,
    print_eval=False,
    print_layer=False,
    batch_size=batch_size,
)


# %%
util.draw_scores(ls_scores_train, "None", hyperparams, "SE Score")

# %%
h_2 = layers_pre_2[1].getPrevOut()

# %% [markdown]
# ### Third FC

# %%
layers_pre_3 = []
layers_pre_3.append(mylayers.InputLayer(h_2, z_score=False))
# ---In---
fc_layer = mylayers.FullyConnectedLayer(64, 10)
layers_pre_3.append(fc_layer)
# ---Out---
fc_layer = mylayers.FullyConnectedLayer(10, 64)
layers_pre_3.append(fc_layer)
# ---valid---
layers_pre_3.append(mylayers.SquaredError())

# %%
hyperparams = {
    "epoch": 10000,
    "eta": 10e-6,
    "batch_size": 1000,
    "rho1": 0.9,
    "rho2": 0.999,
}
epoch = hyperparams["epoch"]
eta = hyperparams["eta"]
batch_size = hyperparams["batch_size"]
rho1 = hyperparams["rho1"]
rho2 = hyperparams["rho2"]
ls_scores_train, Y_hat, acc = test.adam_learn(
    h_2,
    h_2,
    layers_pre_3,
    mylayers.SquaredError,
    epoch=epoch,
    eta=eta,
    print_eval=False,
    print_layer=False,
    batch_size=batch_size,
)


# %%
util.draw_scores(ls_scores_train, "None", hyperparams, "SE Score")

# %%
h_3 = layers_pre_3[1].getPrevOut()

# %% [markdown]
# ## Build layers

# %%
layers = []
# ---Input Layer---
layers.append(mylayers.InputLayer(X_train))
# ---Hidden Layer---
# Greedy pre trained
fc_layer = layers_pre_1[1]
layers.append(fc_layer)
layers.append(mylayers.ReLuLayer())
# Greedy pre trained
fc_layer = layers_pre_2[1]
layers.append(fc_layer)
layers.append(mylayers.ReLuLayer())
# Greedy pre trained
fc_layer = layers_pre_3[1]
layers.append(fc_layer)
layers.append(mylayers.SoftmaxLayer())
# ---Objective---
layers.append(mylayers.CrossEntropy())


# %%
layers

# %% [markdown]
# ### layers test

# %%
hyperparams = {
    "epoch": 500,
    "eta": 10e-6,
    "batch_size": 100,
    "rho1": 0.9,
    "rho2": 0.999,
}
epoch = hyperparams["epoch"]
eta = hyperparams["eta"]
batch_size = hyperparams["batch_size"]
rho1 = hyperparams["rho1"]
rho2 = hyperparams["rho2"]
(
    ls_scores_train,
    Y_hat,
    ls_accuracy_train,
    ls_scores_valid,
    ls_accuracy_valid,
) = test.adam_learn_with_validation(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    layers,
    mylayers.CrossEntropy,
    epoch=epoch,
    eta=eta,
    print_eval=False,
    print_layer=False,
    batch_size=batch_size,
)


# %%
# graph of scores
hidden_layers = "relu -> relu -> softmax"
fig = util.draw_scores_with_validation(
    ls_scores_train, ls_scores_valid, hidden_layers, hyperparams
)
fig.show()


# %%
# graph of accuracys
fig = util.draw_accuracys_with_validation(
    ls_accuracy_train, ls_accuracy_valid, hidden_layers, hyperparams
)
fig.show()


# %% [markdown]
# ### validate accuracy

# %%
labeled_Y_hat = encoder.decoding(Y_hat)
labeled_Y_hat


# %%
labeled_Y = encoder.decoding(Y_train)
labeled_Y


# %%
c = 0
for i in range(len(labeled_Y)):
    if labeled_Y[i] == labeled_Y_hat[i]:
        c += 1
print("Final Accuracy:", (c / len(labeled_Y)) * 100, "%")


# %%
