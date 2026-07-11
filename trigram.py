# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    return F, plt, torch


@app.cell
def _():
    with open("names.txt", "r") as f:
        words = f.read().splitlines()
    return (words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stage 1: Look at frequency of trigrams
    """)
    return


@app.cell
def _(words):
    def sort_trigrams_by_freq():
        b = {}

        for w in words:
            chs = ['<S>'] + list(w) + ['<E>']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                trigram = (ch1, ch2, ch3)
                b[trigram] = b.get(trigram, 0) + 1

        return sorted(b.items(), key = lambda kv: -kv[1])

    sort_trigrams_by_freq()
    return


@app.cell
def _(torch):
    # in a trigram model, we use two chars to predict the third char
    N = torch.zeros((27, 27, 27), dtype=torch.int32)
    return (N,)


@app.cell
def _(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)} # reserve 0 for the start and stop character
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return itos, stoi


@app.cell
def _(N, stoi, words):
    # note there are two start tokens when building a trigram
    def build_trigram_counts():
        for w in words:
            chs = ["."] + ["."] + list(w) + ["."]
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                ix3 = stoi[ch3]
                N[ix1, ix2, ix3] += 1

    # Run the function
    build_trigram_counts()
    return


@app.cell
def _(N, itos, plt, stoi):
    # visualise the trigram counts

    first = stoi['d'] # choose the first char

    plt.figure(figsize=(16, 16))
    plt.imshow(N[first], cmap='Blues')

    for i in range(27):
        for j in range(27):
            chstr = itos[first] + itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, str(N[first, i, j].item()), ha="center", va="top", color="gray")

    plt.axis("off")

    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stage 2: Create a Neural Net
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
    return W, X, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### There should be the same number of trigrams and bigrams

    E.g.,

    #### daniel as a bigram:
    .d, da,an, ni,ie,el, l.

    count: 7

    #### daniel as a trigram:

    ..d, .da, dan, ani, nie, iel, el.

    count: 7
    """)
    return


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
        probs = counts / counts.sum(dim=1, keepdim=True) # probabilities for next character
        return probs

    # gradient descent
    for k in range(1000):
        probs1 = forward_pass(X)
        # extract the probabilities assigned to the correct character and find the negative log loss
        # into probs we feed two 1D series, which are paired up to give for each feature, the column with the correct character in
        loss = -probs1[torch.arange(Y.shape[0]), Y].log().mean() + 0.01 * (W**2).mean()

        # print(loss.item())

        # Backward pass (Calculates gradients and destroys the graph)
        W.grad = None
        loss.backward()

        if W.grad is not None:
            W.data += -50 * W.grad

    '''
    Gemini explanation of why such a big learning rate is optimal:
    Your specific model has no hidden layers, no activation functions, and processes the entire dataset at once (full batch). This creates a loss landscape that is an almost perfectly smooth, incredibly shallow, giant cereal bowl. The gradients (the slopes) are remarkably tiny. Because the slope is practically flat, you need a massive learning rate multiplier to take a step big enough to actually move the weights toward the center of the bowl!
    '''
    return forward_pass, loss


@app.cell
def _(loss):
    print(loss.item())
    return


@app.cell
def _(forward_pass, itos, torch):
    def output_sample():
        g2 = torch.Generator().manual_seed(2147483647)
        for i in range(5):
            out = []
            ix = 0

        context = [0, 0]
        while True:
            probs2 = forward_pass(torch.tensor(context))

            ix = torch.multinomial(
                probs2, num_samples=1, replacement=True, generator=g2
            ).item()
            out.append(itos[int(ix)])
            if ix == 0:
                break
            context = context[1:] + [ix]
        print("".join(out))

    output_sample()
    return


@app.cell
def _(forward_pass, itos, stoi, torch):
    def evaluate_name(name):
        # 1. Initialize context with two start tokens
        context = [0, 0]

        # 2. Append the end token to the name
        name_chars = list(name) + ['.']

        total_log_likelihood = 0.0

        print(f"Evaluating name: '{name}'")
        print("-" * 60)

        for ch in name_chars:
            ix = stoi[ch]

            # Run the forward pass for the current context
            probs = forward_pass(torch.tensor([context]))

            # Get the top 5 predictions ---
            # We use probs[0] to strip the batch dimension and just look at the 27 probabilities
            top_probs, top_ixs = torch.topk(probs[0], 5)

            # Build a readable string of the top 5 characters and their probabilities
            top_5_preds = []
            for p, i in zip(top_probs, top_ixs):
                char = itos[i.item()]
                top_5_preds.append(f"'{char}': {p.item():.4f}")
            top_5_str = " | ".join(top_5_preds)
            # --------------------------------------

            # Pluck out the specific probability for the correct character
            prob = probs[0, ix].item()
            logprob = torch.log(torch.tensor(prob)).item()

            # Print the step (Now including the Top 5!)
            ctx_str = ''.join([itos[c] for c in context])

            print(f"Context: '{ctx_str}' -> Target: '{ch}' (Prob: {prob:.4f}, LogProb: {logprob:.4f})")
            print(f"   Top 5: {top_5_str}")
            print()

            total_log_likelihood += logprob

            # Slide the window
            context = context[1:] + [ix]

        print("-" * 60)
        print(f"Total Log-Likelihood: {total_log_likelihood:.4f}")

        nll = -total_log_likelihood / len(name_chars)
        print(f"Average NLL (Loss): {nll:.4f}")

    evaluate_name('daniel')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
