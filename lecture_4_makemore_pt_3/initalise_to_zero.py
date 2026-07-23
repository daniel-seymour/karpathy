import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt # for making figures

    return F, mo, plt, torch


@app.cell
def _():
    # read in all the words
    words = open('lecture_4_makemore_pt_3/names.txt', 'r').read().splitlines()
    words[:8]
    return (words,)


@app.cell
def _(words):
    len(words)
    return


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
    # layers = [
    #   Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #   Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #   Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
    # ]

    layers = [
      Linear(n_embd * block_size, n_hidden), Tanh(),
      Linear(           n_hidden, n_hidden), Tanh(),
      Linear(           n_hidden, n_hidden), Tanh(),
      Linear(           n_hidden, n_hidden), Tanh(),
      Linear(           n_hidden, n_hidden), Tanh(),
      Linear(           n_hidden, vocab_size),
    ]

    with torch.no_grad():
      # last layer: make less confident for the first training run
      # layers[-1].gamma *= 0.1 # when last layer is a batchnorm
      # layers[-1].weight *= 0.1
      # all other layers: apply gain
      for layer in layers: # initialise all layers to zero
        if isinstance(layer, Linear):
          layer.weight *= 0 # 1.0 for when batchnorm is used

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters)) # number of parameters in total
    for p in parameters:
      p.requires_grad = True
    return C, Tanh, g, layers, parameters


@app.cell
def _(parameters):
    len(parameters)
    return


@app.cell
def _(C, F, Xtr, Ytr, g, layers, parameters, torch):
    def _():
        # same optimization as last time
        max_steps = 200000
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
          lossi.append(loss.log10().item())
          with torch.no_grad():
            ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

          if i >= 1000:
            break # AFTER_DEBUG: would take out obviously to run full optimization

        return ud
    ud = _()
    return (ud,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the uniform loss using NLL is -ln(1/27) = 3.2958 which is higher than when the weights and bias matrices are set to zero implying there is partial training.
    """)
    return


@app.cell
def _(layers):
    last_lin = layers[-1]
    last_lin.bias
    return (last_lin,)


@app.cell
def _(last_lin):
    last_lin.weight
    return


@app.cell
def _(layers):
    first_lin = layers[0]
    first_lin.bias
    return


@app.cell
def _(Tanh, layers, plt, torch):
    def _():
        # visualize histograms
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
          if isinstance(layer, Tanh):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
        plt.legend(legends);
        return plt.title('Tanh/activation output distribution')

    _()
    return


@app.cell
def _(Tanh, layers, plt, torch):
    def _():
        # visualize histograms
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
          if isinstance(layer, Tanh):
            t = layer.out.grad
            print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
        plt.legend(legends);
        return plt.title('Tanh gradient distribution: Check for vanishing gradients')

    _()
    return


@app.cell
def _(parameters, plt, torch):
    def _():
        # visualize histograms
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i,p in enumerate(parameters):
          t = p.grad
          if p.ndim == 2: # only plot the character embeddings and the weight matrices
            print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
        plt.legend(legends)
        return plt.title('Weights gradient distribution: Check for ');

    _()
    return


@app.cell
def _(parameters, plt, ud):
    def _():
        plt.figure(figsize=(20, 4))
        legends = []
        for i,p in enumerate(parameters):
          if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
        plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
        return plt.legend(legends);


    _()
    return


@app.cell
def _(mo):
    import numpy as np
    import scipy.stats as stats

    x0_slider = mo.ui.slider(start=-30, stop=30, step=0.5, value=0.0, label="x0")
    x0_slider
    return np, stats, x0_slider


@app.cell
def _(np, plt, stats, torch, x0_slider):
    def _():
        g = torch.Generator().manual_seed(2147483647+1)
        x = torch.randn(5, generator=g) * 5

        # Read the reactive value directly from the slider
        x[0] = x0_slider.value 

        mu = x.mean()
        sig = x.std()
        y = (x - mu)/sig

        fig, ax = plt.subplots(figsize=(10, 5))

        # plot 0
        ax.plot([-6, 6], [0, 0], 'k')

        # plot the mean and std
        xx = np.linspace(-6, 6, 100)
        ax.plot(xx, stats.norm.pdf(xx, mu, sig), 'b')
        ax.plot(xx, stats.norm.pdf(xx, 0, 1), 'r')

        # plot little lines connecting input and output
        for i in range(len(x)):
            ax.plot([x[i], y[i]], [1, 0], 'k', alpha=0.2)

        # plot the input and output values
        ax.scatter(x.data, torch.ones_like(x).data, c='b', s=100)
        ax.scatter(y.data, torch.zeros_like(y).data, c='r', s=100)
        ax.set_xlim(-6, 6)

        # title
        ax.set_title(f'input mu {mu:.2f} std {sig:.2f}')

        # Output the figure
        return fig


    _()
    return


@app.cell
def _(torch):
    def _():
        # Linear: activation statistics of forward and backward pass

        g = torch.Generator().manual_seed(2147483647)

        a = torch.randn((1000,1), requires_grad=True, generator=g)          # a.grad = b.T @ c.grad
        b = torch.randn((1000,1000), requires_grad=True, generator=g)       # b.grad = c.grad @ a.T
        c = b @ a # (1000, 1000) @ (1000, 1)
        # loss must be scalar
        loss = torch.randn(1000, generator=g) @ c # (1000) @ (1000, 1); (1000) -> (1, 1000). This is not broadcasting, this is a matrix multiplication rule that (1000, ) that if the first arg is 1D then prepend a dimension of size 1  to make it (1, 1000). Whatever operand is 1D (right or left), this is treated as either the col or the row vector to make the matric multiplication work

        # retain grad as normally the gradients of the non-leaf nodes are thrown away
        a.retain_grad()
        b.retain_grad()
        c.retain_grad()
        loss.backward()
        print('a std:', a.std().item())
        print('b std:', b.std().item())
        print('c std:', c.std().item())
        print('-----')
        print('c grad std:', c.grad.std().item())
        print('a grad std:', a.grad.std().item())
        return print('b grad std:', b.grad.std().item())


    _()

    # notice that the c std being 30 implying that the variance is around 1000 which is the fan in
    return


@app.cell
def _(torch):
    def _():
        # Linear + BatchNorm: activation statistics of forward and backward pass

        g = torch.Generator().manual_seed(2147483647)

        n = 1000
        # linear layer ---
        inp = torch.randn(n, requires_grad=True, generator=g)
        w = torch.randn((n, n), requires_grad=True, generator=g) # / n**0.5
        x = w @ inp # (n, n) @ (n, 1) - when the second argument is 1D, PyTorch temporarily treats it as a column vector
        # bn layer ---
        xmean = x.mean()
        xvar = x.var()
        out = (x - xmean) / torch.sqrt(xvar + 1e-5)
        # ----
        loss = out @ torch.randn(n, generator=g)
        inp.retain_grad()
        x.retain_grad()
        w.retain_grad()
        out.retain_grad()
        loss.backward()

        print('inp std: ', inp.std().item())
        print('w std: ', w.std().item())
        print('x std: ', x.std().item())
        print('out std: ', out.std().item())
        print('------')
        print('out grad std: ', out.grad.std().item())
        print('x grad std: ', x.grad.std().item())
        print('w grad std: ', w.grad.std().item())
        return print('inp grad std: ', inp.grad.std().item())


    _()
    return


if __name__ == "__main__":
    app.run()
