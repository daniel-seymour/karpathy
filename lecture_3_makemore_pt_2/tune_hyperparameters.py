import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import marimo as mo

    return F, plt, torch


@app.cell
def _():
    words = open('names.txt', 'r').read().splitlines()
    words[:8]
    return (words,)


@app.cell
def _(words):
    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    print(itos)
    return (stoi,)


@app.cell
def _(stoi, torch, words):
    # --- Hyperparameters ---
    vocab_size = 27 # fixed here as just using characters
    n_embd = 15
    n_hidden1 = 300
    n_hidden2 = 100
    block_size = 3
    # -----------------------

    def build_dataset(words):  
      X, Y = [], []
      for w in words:

        #print(w)
        context = [0] * block_size
        for ch in w + '.':
          ix = stoi[ch]
          X.append(context)
          Y.append(ix)
          #print(''.join(itos[i] for i in context), '--->', itos[ix])
          context = context[1:] + [ix] # crop and append

      X = torch.tensor(X)
      Y = torch.tensor(Y)
      print(X.shape, Y.shape)
      return X, Y

    import random
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])
    return (
        Xdev,
        Xte,
        Xtr,
        Ydev,
        Yte,
        Ytr,
        block_size,
        n_embd,
        n_hidden1,
        n_hidden2,
        vocab_size,
    )


@app.cell
def _(block_size, n_embd, n_hidden1, n_hidden2, torch, vocab_size):
    fan_in_1 = block_size * n_embd

    g = torch.Generator().manual_seed(2147483647) # for reproducibility

    # Embeddings
    C  = torch.randn((vocab_size, n_embd), generator=g)

    # Hidden Layer 1
    W1 = torch.randn((block_size * n_embd, n_hidden1), generator=g)  * (5.0 / 3.0) / fan_in_1**0.5 # Kaiming initalisation
    b1 = torch.randn(n_hidden1, generator=g)

    # Hidden Layer 2
    W2 = torch.randn((n_hidden1, n_hidden2), generator=g) * (5.0 / 3.0) / n_hidden1**0.5 # Kaiming initalisation
    b2 = torch.randn(n_hidden2, generator=g)

    # Output Layer
    W3 = torch.randn((n_hidden2, vocab_size), generator=g) * 0.01 # initalise output probabilities to be uniform zero
    b3 = torch.zeros(vocab_size)  # initalise to zeroes

    parameters = [C, W1, b1, W2, b2, W3, b3]

    for p in parameters:
      p.requires_grad = True

    print(sum(p.nelement() for p in parameters)) # number of parameters in total
    return C, W1, W2, W3, b1, b2, b3, parameters


@app.cell
def _(W1):
    W1
    return


@app.cell
def _(C, F, W1, W2, W3, Xtr, Ytr, b1, b2, b3, parameters, torch):
    lri = []
    lossi = []
    stepi = []

    batch_size = 64

    for i in range(200000):
        # minibatch construct

        ix = torch.randint(0, Xtr.shape[0], (batch_size,))

        # forward pass
        emb = C[Xtr[ix]]  # (batch_size, block_size, n_embd)

        # layer 1
        # emb.shape[0] is batch size
        h1 = torch.tanh(
            emb.view(emb.shape[0], -1) @ W1 + b1
        )  # (batch_size, block_size*n_embds) x (block_size*n_embds, n_hidden)
    
        # layer 2
        h2 = torch.tanh(h1 @ W2 + b2)

        # output layer    
        logits = h2 @ W3 + b3  # (batch_size, vocab_size)
        loss = F.cross_entropy(logits, Ytr[ix])
        # print(loss.item())

        # backward pass
        for p1 in parameters:
            p1.grad = None
        loss.backward()

        # update
        lr = (
            0.1 if i < 50000 else 0.01
        )  # note change from karpathy based off the graph
    
        for p2 in parameters:
            p2.data += -lr * p2.grad

        # track stats
        stepi.append(i)
        lossi.append(loss.log10().item())
    return loss, lossi, stepi


@app.cell
def _(loss):
    print(loss.item())
    return


@app.cell
def _(lossi, plt, stepi):
    plt.plot(stepi, lossi)

    plt.xlabel("Step")

    plt.ylabel("Log Loss") # eg interpret as log_10(0.5) = 10^0.5 = 3.16 (around uniform)

    plt.title("Training Loss")

    plt.show()
    return


@app.cell
def _(C, F, W1, W2, W3, Xdev, Ydev, b1, b2, b3, torch):
    def dev_set():
        # Dev set

        emb = C[Xdev] # (32, 3, 2)
        h1 = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1) 
        h2 = torch.tanh(h1 @ W2 + b2)
        logits = h2 @ W3 + b3 
        loss = F.cross_entropy(logits, Ydev)
        return loss


    dev_set()
    return


@app.cell
def _(C, F, W1, W2, W3, Xte, Yte, b1, b2, b3, torch):
    def test_set():
    
        # Test set
        emb = C[Xte] 
        h1 = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1) 
        h2 = torch.tanh(h1 @ W2 + b2)
        logits = h2 @ W3 + b3 
        loss = F.cross_entropy(logits, Yte)
        return loss


    test_set()
    return


app._unparsable_cell(
    r"""
    def _():
        g = torch.Generator().manual_seed(2147483647 + 10)

            for _ in range(20):
                out = []
                context = [0] * block_size # initialize with all ...
                while True:
                  emb = C[torch.tensor([context])] # (1,block_size,d)
                  h1 = torch.tanh(emb.view(1, -1) @ W1 + b1)
                  h2 = torch.tanh(h1 @ W2 + b2)
                  logits = h2 @ W3 + b3
                  probs = F.softmax(logits, dim=1)
                  ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                  context = context[1:] + [ix]
                  out.append(ix)
                  if ix == 0:
                    break
                print(''.join(itos[i] for i in out))
        
    _()
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
