import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import random
    import marimo as mo
    import matplotlib.pyplot as plt

    return F, mo, plt, random, torch


@app.cell
def _():
    with open("names.txt", "r") as f:
        words = f.read().splitlines()
    return (words,)


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

    def train_trigram(
        train_words, epochs=100, learning_rate=15, reg_lambda=0.05
    ):
        X, Y = create_trigram_training_set(train_words)

        # Initialise the network
        g = torch.Generator().manual_seed(2147483647)
        W = torch.randn((54, 27), generator=g, requires_grad=True)

        loss_history = []
        nll_history = []
        reg_history = []

        # Gradient Descent Loop
        for k in range(epochs):
            # --- Forward pass ---
            xenc = F.one_hot(X, num_classes=27).float()
            xenc_flattened = xenc.view(-1, 54)

            logits = xenc_flattened @ W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            # --- Calculate loss split into components ---

            # Prediction Loss (Negative Log Likelihood)
            nll_loss = -probs[torch.arange(Y.shape[0]), Y].log().mean()

            # Regularization Loss (L2 penalty)
            reg_loss = reg_lambda * (W**2).mean()

            # Total Loss
            loss = nll_loss + reg_loss

            # --- Backward pass ---
            W.grad = None
            loss.backward()

            # --- Update weights ---
            W.data += -learning_rate * W.grad

            loss_history.append(loss.item())
            nll_history.append(nll_loss.item())
            reg_history.append(reg_loss.item())

        # print(f"Final total loss: {loss.item():.4f}")

        # Return the tracked metrics alongside your weights
        return W, loss_history, nll_history, reg_history

    # by trial and error, a learning rate of 36 seems best with no regularisation weight
    W_trigram, total_loss, nll, reg = train_trigram(
        train_words, epochs=100, learning_rate=36, reg_lambda=0.0
    )
    return (
        W_trigram,
        create_trigram_training_set,
        nll,
        reg,
        total_loss,
        train_trigram,
    )


@app.cell
def _(nll, plt, reg, total_loss):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(total_loss, label="Total Loss", linewidth=2, color="blue")
    ax.plot(nll, label="Prediction Loss (NLL)", linestyle="--", color="orange")
    ax.plot(reg, label="Regularization Loss", linestyle=":", color="green")

    ax.set_title("Trigram Model Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig
    return


@app.cell
def _(F, W_trigram, create_trigram_training_set, dev_words, test_words, torch):
    def evaluate_trigram(dataset, W):
            X, Y = create_trigram_training_set(dataset)

            with torch.no_grad(): # for reduced memory and safety net to ensure no training
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
    return (evaluate_trigram,)


@app.cell
def _(dev_words, evaluate_trigram, train_trigram, train_words):
    # Grid Search
    lambda_values = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2]

    dev_loss_results = {}

    # 3. Loop through each value automatically
    for r in lambda_values:
        print(f"Training with reg_lambda = {r}...")
    
        # Train the model
        W, _, _, _ = train_trigram(
            train_words, 
            epochs=100, 
            learning_rate=36, 
            reg_lambda=r
        )
    
        # Evaluate immediately on the Dev Set
        dev_loss = evaluate_trigram(dev_words, W)
    
        dev_loss_results[r] = dev_loss
        print(f"--> Dev Loss: {dev_loss:.4f}\n")

    # This finds the dictionary key (the lambda) that has the lowest associated value (the loss)
    best_lambda = min(dev_loss_results, key=dev_loss_results.get)

    print("="*30)
    print(f"Best Regularization Weight: {best_lambda}")
    print(f"Lowest Dev Loss: {dev_loss_results[best_lambda]:.4f}")
    print("="*30)
    return


@app.cell
def _(evaluate_trigram, test_words, train_trigram, train_words):
    W_final, *_ = train_trigram(
        train_words, epochs=100, learning_rate=36, reg_lambda=0.0
    )

    test_loss = evaluate_trigram(test_words, W_final)
    print(f"{test_loss:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Findings

    The trigram model has the lowest loss on dev when there is zero weight on regularisation.

    I also found that a large learning rate and a large regularisation weight leads to gradient explosion because the derivative of the regularisation expression shows up in the gradient and can lead to enormous steps.
    """)
    return


if __name__ == "__main__":
    app.run()
