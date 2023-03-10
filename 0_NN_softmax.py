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


# %%
df_train.shape

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


# %% [markdown]
# ## Build layers

# %%
layers = []
# ---Input Layer---
layers.append(mylayers.InputLayer(X_train))
# ---Hidden Layer---
# initialize FCL with xaiver
fc_layer = mylayers.FullyConnectedLayer(X_train.shape[1], 10)
weights, biases = util.xaiver_init(X_train.shape[1], 10)
fc_layer.setWeights(weights)
fc_layer.setBiases(biases)
layers.append(fc_layer)
layers.append(mylayers.SoftmaxLayer())
# ---Output Layer---
# fc_layer = mylayers.FullyConnectedLayer(20, 10)
# weights, biases = util.xaiver_init(20, 10)
# fc_layer.setWeights(weights)
# fc_layer.setBiases(biases)
# layers.append(fc_layer)
# layers.append(mylayers.SoftmaxLayer())
# ---Objective---
layers.append(mylayers.CrossEntropy())


# %% [markdown]
# ### layers test

# %%
hyperparams = {"epoch": 500, "eta": 10e-8, "batch_size": 5, "rho1": 0.9, "rho2": 0.999}


# %%
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
hidden_layers = "softmax"
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
