import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import marimo as mo

    return F, mo, torch


@app.cell
def _():
    with open("names.txt", "r") as f:
        words = f.read().splitlines()
    return (words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split dataset
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Bigram
    """)
    return


@app.cell
def _(stoi, torch, words):
    # create the dataset
    xs, ys = [], []
    for w in words:
      chs = ['.'] + list(w) + ['.']
      for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print('number of examples: ', num)

    # initialize the 'network'
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    return W, num, xs, ys


@app.cell
def _(F, W, num, torch, xs, ys):
    # gradient descent
    for k in range(1):

      # forward pass
      xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
      logits = xenc @ W # predict log-counts
      counts = logits.exp() # counts, equivalent to N
      probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
      loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
      print(loss.item())

      # backward pass
      W.grad = None # set to zero the gradient
      loss.backward()

      # update
      W.data += -50 * W.grad
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Trigram
    """)
    return


@app.cell
def _(stoi, torch, words):
    # create the training set of trigrams (x,y, z)

    def create_training_set():
        X, Y = [], []
        for w in words:
            chs = ["."] + ["."] + list(w) + ["."]
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]

                X.append([ix1, ix2])
                Y.append(ix3)

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

    X, Y = create_training_set()
    num = X.shape[0] # should not use nelement here as it will count each element
    print('Number of examples: ', num)

    # Initalise the network
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((54, 27), generator=g, requires_grad=True)
    return W, X, Y, num


@app.cell
def _(F, W, X, Y, torch):
    def forward_pass(x_input):
        # 1. X has shape [N, 2] (e.g., two integers per row)
        # 2. xenc gets shape [N, 2, 27]
        xenc = F.one_hot(x_input, num_classes=27).float()

        # 3. Reshape the contiguous memory into a 2D matrix of shape [N, 54]
        xenc_flattened = xenc.view(-1, 54)

        # Forward pass
        # xenc_flattened: n x 54, W: 54 x 27
        logits = xenc_flattened @ W # n x 27
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
        return probs

    # gradient descent
    for k in range(1000):
        probs1 = forward_pass(X)
        # extract the probabilities assigned to the correct character and find the negative log loss
        # into probs we feed two 1D series, which are paired up to give for each feature, the column with the correct character in
        loss = -probs1[torch.arange(Y.shape[0]), Y].log().mean() + 0.01*(W**2).mean()

        # print(loss.item())

        # Backward pass (Calculates gradients and destroys the graph)
        W.grad = None
        loss.backward()

        W.data += -50 * W.grad
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate and compare the bigram and trigram models
    """)
    return


if __name__ == "__main__":
    app.run()
