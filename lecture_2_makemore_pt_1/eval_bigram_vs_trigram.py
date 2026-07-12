import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import random
    import marimo as mo

    return F, mo, random, torch


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
def _(random, words):
    random.seed(42)
    random.shuffle(words)

    total_words = len(words)
    n1 = int(0.8 * total_words)
    n2 = int(0.9 * total_words)

    train_words = words[:n1]
    dev_words = words[n1:n2]
    test_words = words[n2:]

    print(f"Total: {total_words}")
    print(f"Train: {len(train_words)} (80%)")
    print(f"Dev:   {len(dev_words)} (10%)")
    print(f"Test:  {len(test_words)} (10%)")
    return dev_words, test_words, train_words


@app.cell
def _(words):
    chars = sorted(list(set("".join(words))))
    stoi = {
        s: i + 1 for i, s in enumerate(chars)
    }  # reserve 0 for the start and stop character
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return (stoi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Bigram
    """)
    return


@app.cell
def _(F, stoi, torch, train_words):
    def create_bigram_training_set(words):
        X, Y = [], []
        for w in words:
            # Pad with one dot each side for the bigram context
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]

                X.append(ix1)
                Y.append(ix2)

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        num = X.shape[0]
        print(f"Number of examples: {num}")
        return X, Y

    def train_bigram(train_words):
        X, Y = create_bigram_training_set(train_words)

        # Initialise the network
        g = torch.Generator().manual_seed(2147483647)
        # 27 inputs (1 character * 27 dimensions), 27 outputs (vocab size)
        W = torch.randn((27, 27), generator=g, requires_grad=True)

        # Gradient Descent Loop
        epochs = 1
        for k in range(epochs):
            # --- Forward pass ---

            # One hot encoding
            # Convert X to one-hot encoding
            xenc = F.one_hot(X, num_classes=27).float()

            # Matrix multiplication: n x 27 @ 27 x 27 = n x 27
            logits = xenc @ W

            # ------------------
            # # Alternative without use of one hot encoding
            # logits_alt = torch.zeros(X.shape[0], 27)

            # for i in range(X.shape[0]):
            #     for j in range(W.shape[1]): # iterate through cols of weight matrix
            #         logits_alt[i,j] = W[X[i],j] # go to the row in the weights matrix dictated by the element in X

            # print(logits_alt)
            # print(torch.allclose(logits, logits_alt))
            # ------------------

            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            # --- Calculate Loss ---
            # Negative log likelihood + L2 regularization
            loss = (
                -probs[torch.arange(Y.shape[0]), Y].log().mean()
                + 0.01 * (W**2).mean()
            )

            # ------------------
            # Alternative loss calculation using F.cross_entropy
            loss_alt = F.cross_entropy(logits, Y) + 0.01 * (W**2).mean()

            print(loss_alt.item())
            print(torch.isclose(loss, loss_alt).item())
            # ------------------

            # --- Backward pass ---
            W.grad = None
            loss.backward()

            # --- Update weights ---
            W.data += -50 * W.grad

        print(f"Final loss after {epochs} epochs: {loss.item():.4f}")

        return W

    W_bigram = train_bigram(train_words)
    return W_bigram, create_bigram_training_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Trigram
    """)
    return


@app.cell
def _(F, stoi, torch, train_words):
    def create_trigram_training_set(words):
        X, Y = [], []
        for w in words:
            # Pad with two dots for the trigram context
            chs = ["."] + ["."] + list(w) + ["."]
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]

                X.append([ix1, ix2])
                Y.append(ix3)

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        num = X.shape[0]
        print(f"Number of examples: {num}")

        return X, Y

    def train_trigram(train_words):
        X, Y = create_trigram_training_set(train_words)

        # Initialise the network
        g = torch.Generator().manual_seed(2147483647)
        # 54 inputs (2 characters * 27 dimensions), 27 outputs (vocab size)
        W = torch.randn((54, 27), generator=g, requires_grad=True)

        # Gradient Descent Loop
        epochs = 100
        for k in range(epochs):
            # --- Forward pass ---
            # Convert X to one-hot encoding and flatten from [N, 2, 27] to [N, 54]
            xenc = F.one_hot(X, num_classes=27).float()
            xenc_flattened = xenc.view(-1, 54)

            # Matrix multiplication: n x 54 @ 54 x 27 = n x 27
            logits = xenc_flattened @ W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            # --- Calculate Loss ---
            # Negative log likelihood + L2 regularization
            loss = (
                -probs[torch.arange(Y.shape[0]), Y].log().mean()
                + 0.01 * (W**2).mean()
            )

            # --- Backward pass ---
            W.grad = None
            loss.backward()

            # --- Update weights ---
            W.data += -50 * W.grad

        print(f"Final loss after {epochs} epochs: {loss.item():.4f}")

        return W

    W_trigram = train_trigram(train_words)
    return W_trigram, create_trigram_training_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate and compare the bigram and trigram models on the dev and test splits
    """)
    return


@app.cell
def _(F, W_bigram, create_bigram_training_set, dev_words, test_words, torch):
    def evaluate_bigram(dataset, W):
        X, Y = create_bigram_training_set(dataset)

        with (
            torch.no_grad()
        ):  # for reduced memory and safety net to ensure no training
            # Forward pass
            xenc = F.one_hot(X, num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            # Calculate the loss
            loss = -probs[torch.arange(Y.shape[0]), Y].log().mean()

            print(f"Evaluation loss: {loss.item():.4f}")

        return loss.item()

    loss_bigram_dev = evaluate_bigram(dev_words, W_bigram)
    loss_bigram_test = evaluate_bigram(test_words, W_bigram)
    return loss_bigram_dev, loss_bigram_test


@app.cell
def _(F, W_trigram, create_trigram_training_set, dev_words, test_words, torch):
    def evaluate_trigram(dataset, W):
        X, Y = create_trigram_training_set(dataset)

        with (
            torch.no_grad()
        ):  # for reduced memory and safety net to ensure no training
            # Forward pass
            xenc = F.one_hot(X, num_classes=27).float()
            xenc_flattened = xenc.view(-1, 54)

            logits = xenc_flattened @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            # Calculate the loss
            loss = -probs[torch.arange(Y.shape[0]), Y].log().mean()

            print(f"Evaluation loss: {loss.item():.4f}")

        return loss.item()

    loss_trigram_dev = evaluate_trigram(dev_words, W_trigram)
    loss_trigram_test = evaluate_trigram(test_words, W_trigram)
    return loss_trigram_dev, loss_trigram_test


@app.cell
def _(
    loss_bigram_dev,
    loss_bigram_test,
    loss_trigram_dev,
    loss_trigram_test,
    mo,
):
    mo.md(f"""
    ### Bigram vs Trigram evaluation

    | Model   | Dev | Test |
    |---------|-----|------|
    | Bigram  | {loss_bigram_dev:.4f} | {loss_bigram_test:.4f} |
    | Trigram | {loss_trigram_dev:.4f} | {loss_trigram_test:.4f} |
    """)
    return


if __name__ == "__main__":
    app.run()
