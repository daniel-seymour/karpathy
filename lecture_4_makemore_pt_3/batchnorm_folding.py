import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt # for making figures

    return F, mo, torch


@app.cell
def _():
    # read in all the words
    words = open('lecture_4_makemore_pt_3/names.txt', 'r').read().splitlines()
    words[:8]
    return (words,)


@app.cell
def _(words):
    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    print(itos)
    print(vocab_size)
    return stoi, vocab_size


@app.cell
def _(stoi, torch, words):
    # build the dataset
    block_size = 3 # context length: how many characters do we take to predict the next one?

    def build_dataset(words):  
      X, Y = [], []

      for w in words:
        context = [0] * block_size
        for ch in w + '.':
          ix = stoi[ch]
          X.append(context)
          Y.append(ix)
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

    Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
    Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
    Xte,  Yte  = build_dataset(words[n2:])     # 10%
    return Xtr, Ytr, block_size


@app.cell
def _(block_size, torch, vocab_size):
    # Let's train a deeper network
    # The classes we create here are the same API as nn.Module in PyTorch

    class Linear:

      def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

      def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
          self.out += self.bias
        return self.out

      def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


    class BatchNorm1d:

      def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

      def __call__(self, x):
        # calculate the forward pass
        if self.training:
          xmean = x.mean(0, keepdim=True) # batch mean
          xvar = x.var(0, keepdim=True) # batch variance
        else:
          xmean = self.running_mean
          xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
          with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

      def parameters(self):
        return [self.gamma, self.beta]

    class Tanh:
      def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
      def parameters(self):
        return []

    n_embd = 10 # the dimensionality of the character embedding vectors
    n_hidden = 100 # the number of neurons in the hidden layer of the MLP
    g = torch.Generator().manual_seed(2147483647) # for reproducibility

    C = torch.randn((vocab_size, n_embd),            generator=g)

    layers = [
      Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
      Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
      Linear(           n_hidden, n_hidden, bias=False)
    ]

    with torch.no_grad():
      # last layer: make less confident for the first training run
      # layers[-1].gamma *= 0.1 # when last layer is a batchnorm
      layers[-1].weight *= 0.1
      # all other layers: apply gain
      for layer in layers[:-1]: # 
        if isinstance(layer, Linear):
          layer.weight *= 1.0 #for when batchnorm is used

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters)) # number of parameters in total
    for p in parameters:
      p.requires_grad = True
    return C, Linear, g, layers, n_embd, n_hidden, parameters


@app.cell
def _(C, F, Xtr, Ytr, g, layers, parameters, torch):
    def _():
        # same optimization as last time
        max_steps = 20000
        batch_size = 32
        lossi = []
        ud = []

        for i in range(max_steps):

          # minibatch construct
          ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
          Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

          # forward pass
          emb = C[Xb] # embed the characters into vectors
          x = emb.view(emb.shape[0], -1) # concatenate the vectors
          for layer in layers:
            x = layer(x)
          loss = F.cross_entropy(x, Yb) # loss function

          # backward pass
          for layer in layers:
            layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
          for p in parameters:
            p.grad = None
          loss.backward()

          # update
          lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
          for p in parameters:
            p.data += -lr * p.grad

          # track stats
          if i % 10000 == 0: # print every once in a while
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
          # lossi.append(loss.log10().item())
          # with torch.no_grad():
          #       return ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

          # if i >= 1000:
          #   break # AFTER_DEBUG: would take out obviously to run full optimization

        return loss


    loss = _()
    print(f"Final: {loss.item():.4f}")
    return


@app.cell
def _(layers):
    def _():
        for layer in layers:
            layer.training = False
        return


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Maths for how to fold in the BatchNorm beta and gamma
    We have an intial linear layer: $y = xW + b$

    At inference time, the BatchNorm layer uses the running mean and variance and looks like:

    $$
    z = \gamma \left(\frac{y-\mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta
    $$

    We now substitute the linear layer into the BatchNorm:

    $$
    z = \gamma \left(\frac{xW+b-\mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta
    $$

    To fold this, we want to manipulate the equation so it looks like a new linear layer:

    $$
    z = xW_{\text{folded}} + b_{\text{folded}}
    $$

    We first isolate the scaling factor into a single term, S:

    $$S = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$‬

    This gives us:
    $$z = S(xW+b-\mu)+\beta$$
    $$z = x(SW) + S(b-\mu)+\beta$$

    so $W_{\text{folded}}=Sw$ and $b_{\text{folded}}=S(b-\mu)+\beta$

    Now we have $W_{\text{folded}}$ and $b_{\text{folded}}$, we can delete the BatchNorm layer from inference because it's 'knowledge' is folded into the linear weights. This also has the advantage of speeding up inference because none of the BatchNorm operations are needed.
    """)
    return


@app.cell
def _(layers, torch):
    # calculate the new folded in weights and biases matrices 

    lin_layer = layers[0]
    bn_layer = layers[1]

    W = lin_layer.weight
    # Linear layers have bias=False, so b is 0 here
    b = lin_layer.bias if lin_layer.bias is not None else torch.zeros_like(bn_layer.running_mean) 


    gamma = bn_layer.gamma
    beta = bn_layer.beta
    mu = bn_layer.running_mean
    var = bn_layer.running_var
    eps = bn_layer.eps

    with torch.no_grad():
        S = gamma / torch.sqrt(var + eps)
        W_folded = S*W
        b_folded = S*(b-mu) + beta
    return W_folded, b_folded, bn_layer, lin_layer, var


@app.cell
def _(var):
    var
    return


@app.cell
def _(
    Linear,
    W_folded,
    b_folded,
    block_size,
    bn_layer,
    lin_layer,
    n_embd,
    n_hidden,
    torch,
):
    # Test whether the folded in weights and biases gives the same answer as the original two step forward pass

    # Dummy input
    x_dummy = torch.randn(1, n_embd * block_size)

    # Force the layers into inference mode to prevent Marimo race conditions
    bn_layer.training = False

    # Option 1: The original two-step forward pass
    with torch.no_grad():
        out_original = bn_layer(lin_layer(x_dummy))

    # Option 2
    # Create a new Linear layer with the folded parameters
    folded_layer = Linear(n_embd * block_size, n_hidden, bias=True)
    folded_layer.weight = W_folded
    folded_layer.bias = b_folded

    # The new one-step forward pass
    with torch.no_grad():
        out_folded = folded_layer(x_dummy)

    # 4. Compare them
    print("Are the outputs identical?", torch.allclose(out_original, out_folded, atol=1e-5))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
