import numpy as np
import plotly.graph_objects as go
import plotly

np.random.RandomState(42)


def xaiver_init(m, n):
    """_summary_

    Args:
        m (_type_): input shape of layer
        n (_type_): output shpae of layer

    Returns:
        _type_: a initial weight matrix and bias with xaiver
    """
    lim = np.sqrt(6 / (m + n))
    weights = np.random.uniform(-lim, lim, (m, n))

    return weights, 0


class onehot_encoder:
    def __init__(self, data):
        self.all_elements = set(data)
        self.element_to_int = dict((c, i) for i, c in enumerate(self.all_elements))
        self.int_to_element = dict((i, c) for i, c in enumerate(self.all_elements))
        self.integer_encoded = [self.element_to_int[ele] for ele in data]

    def encoding(self, df=None):
        if df is None:
            onehot_encoded = list()
            # print(self.int_to_element)
            for i in self.integer_encoded:
                element = [0 for _ in range(len(self.all_elements))]
                element[i] = 1
                onehot_encoded.append(element)
            onehot_encoded = np.array(onehot_encoded)
        else:
            onehot_encoded = list()
            integer_encoded = [self.element_to_int[ele] for ele in df]
            for i in integer_encoded:
                element = [0 for _ in range(len(self.all_elements))]
                element[i] = 1
                onehot_encoded.append(element)
            onehot_encoded = np.array(onehot_encoded)
        return onehot_encoded

    def decoding(self, encoded):
        decoded = []
        for row in encoded:
            element_int = row.argmax()
            decoded.append(self.int_to_element[element_int])
        return decoded


def z_score(data):
    """_summary_

    Args:
        data (_type_): input data

    Returns:
        _type_: z-score normalized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scored_data = (data - mean) / std
    return z_scored_data


def drop_zero_std_columns(df):
    stds = df.std(axis=0)
    zero_std_cols = stds[stds == 0].index
    df = df.drop(columns=zero_std_cols)

    return df


def draw_scores(ls_scores, hidden_layers, hyperparams, score_name="Score"):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(zip(*ls_scores))[0],
            y=list(zip(*ls_scores))[1],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="train score",
        )
    )

    title_text = f"Score to epoch <br>  epoch: {hyperparams['epoch']}, eta: {hyperparams['eta']}, batch size: {hyperparams['batch_size']}, rho1: {hyperparams['rho1']}, rho2: {hyperparams['rho2']} <br>  with hidden layers: {hidden_layers}"
    fig.update_layout(
        title=title_text,
        xaxis_title="Epoch",
        yaxis_title=score_name,
        hovermode="x unified",
    )
    # fig.show()
    return fig


def draw_scores_with_validation(
    ls_scores_train, ls_scores_valid, hidden_layers, hyperparams
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(zip(*ls_scores_train))[0],
            y=list(zip(*ls_scores_train))[1],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="train score",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(zip(*ls_scores_valid))[0],
            y=list(zip(*ls_scores_valid))[1],
            mode="lines",
            line=dict(width=2, color="red"),
            name="valid score",
        )
    )

    title_text = f"Score to epoch <br>  epoch: {hyperparams['epoch']}, eta: {hyperparams['eta']}, batch size: {hyperparams['batch_size']}, rho1: {hyperparams['rho1']}, rho2: {hyperparams['rho2']} <br>  layers: {hidden_layers}"
    fig.update_layout(
        title=title_text,
        xaxis_title="Epoch",
        yaxis_title="Cross Entropy Score",
        hovermode="x unified",
    )
    # fig.show()
    return fig


def draw_accuracys_with_validation(
    ls_accuracy_train, ls_accuracy_valid, hidden_layers, hyperparams
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(zip(*ls_accuracy_train))[0],
            y=list(zip(*ls_accuracy_train))[1],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="train accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(zip(*ls_accuracy_valid))[0],
            y=list(zip(*ls_accuracy_valid))[1],
            mode="lines",
            line=dict(width=2, color="red"),
            name="valid accuracy",
        )
    )

    # fig = px.line(x=list(zip(*ls_accuracy))[0], y=list(zip(*ls_accuracy))[1], )
    title_text = f"Accuracy to epoch <br>  epoch: {hyperparams['epoch']}, eta: {hyperparams['eta']}, batch size: {hyperparams['batch_size']}, rho1: {hyperparams['rho1']}, rho2: {hyperparams['rho2']} <br>  with hidden layers: {hidden_layers}"
    fig.update_layout(
        title=title_text,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified",
    )
    last_epoch = list(zip(*ls_accuracy_train))[0][-1]
    last_acc_train = round(list(zip(*ls_accuracy_train))[1][-1], 2)
    last_point_text_train = (
        f"train accuracy<br>  epoch:{last_epoch+1}<br>  acc: {last_acc_train}]"
    )
    fig.add_annotation(
        x=last_epoch,
        y=list(zip(*ls_accuracy_train))[1][-1],
        xref="x",
        yref="y",
        text=last_point_text_train,
        showarrow=True,
        arrowhead=1,
        font=dict(
            # family="Courier New, monospace",
            size=8,
        ),
        align="center",
        bgcolor="#ffffff",
        opacity=0.7,
    )

    last_acc_valid = round(list(zip(*ls_accuracy_valid))[1][-1], 2)
    last_point_text_valid = (
        f"valid accuracy<br>  epoch:{last_epoch+1}<br>  acc: {last_acc_valid}]"
    )
    fig.add_annotation(
        x=last_epoch,
        y=list(zip(*ls_accuracy_valid))[1][-1],
        xref="x",
        yref="y",
        text=last_point_text_valid,
        showarrow=True,
        arrowhead=1,
        ay=30,
        font=dict(
            # family="Courier New, monospace",
            size=8,
        ),
        align="center",
        bgcolor="#ffffff",
        opacity=0.7,
    )
    return fig
