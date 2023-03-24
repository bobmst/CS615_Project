from tqdm import tqdm
import numpy as np
import mylayers
import asyncio


def layers_forward(h, Y, layers, eval_obj, print_layer=True):
    """_summary_

    Args:
        h (np.array): data in
        Y (np.array): expected output
        layers (list): list of all layers including the objection
        eval_obj (mylayer.layer): _description_
        print_layer (bool, optional): If the name and detail of each layer is going to be printed. Defaults to True.

    Returns:
        (list,list): a list of all socres on each epoch, and a list of predicted y of each epoch
    """
    score = float("inf")
    for i in range(len(layers)):
        if isinstance(layers[i], eval_obj):
            # print(layers[i],Y,h)
            score = layers[i].eval(Y, h)

        else:
            h = layers[i].forward(h)
            if print_layer:
                print("--- Layer", i + 1, "---")
                print(h.shape)
                # print(h)

    return score, h


def layers_fit(h, Y, layers, eval_obj, print_layer=True):

    score = float("inf")
    for i in range(len(layers)):
        if isinstance(layers[i], eval_obj):
            # print(layers[i],Y,h)
            score = layers[i].eval(Y, h)
        else:
            h = layers[i].fit(h)
            if print_layer:
                print("--- Layer", i + 1, "---")
                print(h.shape)

    return score, h


# def layers_backward(h, Y, layers, eval_obj, eta=10e-4, print_layer=True):
#     grad = layers[-1].gradient(Y, h)
#     for i in range(len(layers) - 2, 0, -1):
#         newgrad = layers[i].backward(grad)
#         if isinstance(layers[i], mylayers.FullyConnectedLayer):
#             layers[i].updateWeights(grad, eta)
#         if print_layer:
#             print("--- Layer", i, "---")
#             print(newgrad.shape)
#             # print(newgrad)
#         grad = newgrad
#     return grad


def layers_backward_adam(
    h,
    Y,
    layers,
    eval_obj,
    epoch,
    eta=10e-4,
    print_layer=True,
    rho1=0.9,
    rho2=0.999,
    epsilon=10e-8,
):
    grad = layers[-1].gradient(Y, h)
    for i in range(len(layers) - 2, 0, -1):
        newgrad = layers[i].backward(grad)
        if isinstance(layers[i], mylayers.FullyConnectedLayer):
            # layers[i].updateWeights(grad, eta)
            layers[i].updateWeightsAdam(
                grad,
                epoch,
                eta=eta,
                rho1=rho1,
                rho2=rho2,
                epsilon=epsilon,
            )
        # if isinstance(layers[i], mylayers.ConvolutionLayer):
        #     layers[i].updateKernels(grad, eta=eta)
        if print_layer:
            print("--- Layer", i, "---")
            print(newgrad.shape)
            # print(newgrad)
        grad = newgrad
    return grad


# def learn(
#     X,
#     Y,
#     layers,
#     eval_obj,
#     tolerance=10e-10,
#     epoch=10000,
#     eta=10e-4,
#     print_eval=True,
#     print_layer=False,
#     batch_size=100,
# ):
#     scores = []
#     n_batchs = X.shape[0] // batch_size

#     n_batchs = int(np.ceil(len(X) / batch_size))
#     X_batches = np.array_split(X, n_batchs)
#     Y_batches = np.array_split(Y, n_batchs)

#     y_hat = None
#     # 🗒️if you don't want to use progress bar use:
#     # for e in range(epoch):
#     for e in (pbar := tqdm(range(epoch))):
#         for n_batch in range(n_batchs):
#             mini_X = X_batches[n_batch]
#             mini_Y = Y_batches[n_batch]
#             score, mini_y_hat = layers_forward(
#                 mini_X, mini_Y, layers, eval_obj, print_layer=False
#             )
#             if print_eval:
#                 # 🗒️if you don't want to use progress bar use:
#                 # print(f"=== Epoch {e} ===")
#                 # print(f"🎢Score: {score}")
#                 pbar.set_description(f"♾️Epoch {e} with 🎢Score {round(score, 6)}")
#             new_grad = layers_backward(
#                 mini_y_hat, mini_Y, layers, eval_obj, eta=eta, print_layer=False
#             )
#             if y_hat is not None:
#                 y_hat = np.concatenate((y_hat, mini_y_hat), axis=0)
#             else:
#                 y_hat = mini_y_hat
#         scores.append((e, score))

#         # stop when score is below tolerance
#         if score < tolerance:
#             break
#     return scores, y_hat


def evaluate_accuracy_with_one_hot(Y, Y_hat):
    # labeled_Y_hat = np.array(encoder.decoding(Y_hat))
    # labeled_Y = np.array(encoder.decoding(Y))
    Y_hat = np.argmax(Y_hat, axis=1)
    Y = np.argmax(Y, axis=1)
    return np.sum(Y == Y_hat) / len(Y) * 100.0


def adam_learn(
    X,
    Y,
    layers,
    eval_obj,
    epoch=10000,
    eta=10e-4,
    print_eval=True,
    print_layer=False,
    batch_size=100,
):
    scores = []
    accuracies = []
    n_batchs = X.shape[0] // batch_size

    n_batchs = int(np.ceil(len(X) / batch_size))
    X_batches = np.array_split(X, n_batchs)
    Y_batches = np.array_split(Y, n_batchs)

    # 🗒️if you don't want to use progress bar use:
    # for e in range(epoch):
    for e in (pbar := tqdm(range(epoch))):
        y_hat = None
        for n_batch in range(n_batchs):
            mini_X = X_batches[n_batch]
            mini_Y = Y_batches[n_batch]
            score, mini_y_hat = layers_forward(
                mini_X, mini_Y, layers, eval_obj, print_layer=print_layer
            )
            if print_eval:
                # 🗒️if you don't want to use progress bar use:
                # print(f"=== Epoch {e} ===")
                # print(f"🎢Score: {score}")
                pbar.set_description(f"♾️Epoch {e} with 🎢Score {round(score, 6)}")
            new_grad = layers_backward_adam(
                mini_y_hat,
                mini_Y,
                layers,
                eval_obj,
                e,
                eta=eta,
                print_layer=print_layer,
            )
            if y_hat is not None:
                y_hat = np.concatenate((y_hat, mini_y_hat), axis=0)
            else:
                y_hat = mini_y_hat

        acc = evaluate_accuracy_with_one_hot(Y, y_hat)
        scores.append((e, score))
        accuracies.append((e, acc))

    return scores, y_hat, accuracies


def adam_learn_with_validation(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    layers,
    eval_obj,
    epoch=10000,
    eta=10e-4,
    print_eval=True,
    print_layer=False,
    batch_size=100,
):
    scores_train = []
    accuracies_train = []
    scores_valid = []
    accuracies_valid = []
    n_batchs = X_train.shape[0] // batch_size

    n_batchs = int(np.ceil(len(X_train) / batch_size))
    X_batches = np.array_split(X_train, n_batchs)
    Y_batches = np.array_split(Y_train, n_batchs)

    # 🗒️if you don't want to use progress bar use:
    # for e in range(epoch):
    for e in (pbar := tqdm(range(epoch))):
        y_hat_train = None
        # proform mini batch training
        for n_batch in range(n_batchs):
            mini_X = X_batches[n_batch]
            mini_Y = Y_batches[n_batch]
            score_train, mini_y_hat = layers_forward(
                mini_X, mini_Y, layers, eval_obj, print_layer=print_layer
            )
            if print_eval:
                # 🗒️if you don't want to use progress bar use:
                # print(f"=== Epoch {e} ===")
                # print(f"🎢Score: {score}")
                pbar.set_description(f"♾️Epoch {e} with 🎢Score {round(score, 6)}")

            new_grad = layers_backward_adam(
                mini_y_hat,
                mini_Y,
                layers,
                eval_obj,
                e,
                eta=eta,
                print_layer=print_layer,
            )
            if y_hat_train is not None:
                y_hat_train = np.concatenate((y_hat_train, mini_y_hat), axis=0)
            else:
                y_hat_train = mini_y_hat

        score_valid, y_hat_valid = layers_fit(
            X_valid, Y_valid, layers, eval_obj, print_layer=print_layer
        )

        # append training info
        scores_train.append((e, score_train))

        acc_train = evaluate_accuracy_with_one_hot(Y_train, y_hat_train)
        accuracies_train.append((e, acc_train))

        # append validation info
        scores_valid.append((e, score_valid))

        acc_valid = evaluate_accuracy_with_one_hot(Y_valid, y_hat_valid)
        accuracies_valid.append((e, acc_valid))

    return scores_train, y_hat_train, accuracies_train, scores_valid, accuracies_valid
