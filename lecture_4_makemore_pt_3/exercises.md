# Karpathy Makemore Pt3 Exercises

> E01: I did not get around to seeing what happens when you initialize all weights and biases to zero. Try this and train the neural net. You might think either that 1) the network trains just fine or 2) the network doesn't train at all, but actually it is 3) the network trains but only partially, and achieves a pretty bad final performance. Inspect the gradients and activations to figure out what is happening and why the network is only partially training, and what part is being trained exactly.

**Answer with mathematical justifications and after a more careful look:**

There is partial learning because the loss is below the loss if each character had a constant uniform probability.

We see that in the last layer, the weights are all zeroes, but the biases are not. In all previous layers, the weights and the biases are both all zeroes, because these previous layers depend on the weights matrix of the last layer not being zero for backpropogation. We can see why this is by looking at the gradients analytically:

We have $logits = h W + b$ and $\frac{\partial L}{\partial logits} = \text{probs} - \text{targets}$. This means the partial derivative of the logits wrt to the loss is non-zero.

However, the partial derivative of the weights wrt to the loss is zero because h will be zero.$$\frac{\partial L}{\partial W}= h^T \cdot \frac{\partial L}{\partial \mathrm{logits}}$$

However, the gradient for the biases will be non-zero and therefore the biases will learn. This is because:

$$\frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial logits}$$

The $\sum \frac{\partial L}{\partial logits}$ arises because in the forward pass the 1D bias vector is broadcast so it applies to all the training examples in the batch. As there are 32 examples in the batch size, this means the bias contributes to 32 distinct calculations of the loss. The partial derivative of the logits wrt the bias is just one so it is not written out.

In calculus, the multivariate chain rule states that if a single variable branches out and affects an output through multiple different paths, the total derivative is the sum of the derivatives from every path. This means there must be a summation in the expression of the partial derivative of the bias.

As only the biases are learning, this has the effect of turning the model into a unigram model because the biases are learning the unigram distribution of characters in the training data.

> E02: BatchNorm, unlike other normalization layers like LayerNorm/GroupNorm etc. has the big advantage that after training, the batchnorm gamma/beta can be "folded into" the weights of the preceeding Linear layers, effectively erasing the need to forward it at test time. Set up a small 3-layer MLP with batchnorms, train the network, then "fold" the batchnorm gamma/beta into the preceeding Linear layer's W,b by creating a new W2, b2 and erasing the batch norm. Verify that this gives the same forward pass during inference. i.e. we see that the batchnorm is there just for stabilizing the training, and can be thrown out after training is done! pretty cool.

Verified!

------------------------------------------------------------------------
**Further info that applies to E01:**

The logits matrix is of size (batch_size, vocab_size) where the hidden activations (h) are size (batch_size, num_nuerons) and the weights matrix is of size (num_nuerons, vocab_size). The logits matrix expresses for each training example, the probability of each character being next.

The bias is a flat 1D vector of length vocab_size. It adds a base score to each column, i.e., for each character that could come next, it adds a systematic bias. So for 'a', it does +2.0 and for 'z', it does -8.0. This is applied uniformly to all training examples which are the rows.

------------------------------------------------------------------------

Original answer to E01:

If I initialise all weights and biases including the output layer to zero, then we see that the tanh outputs, the tanh gradients and the weights gradients are all zeroes. Therefore, no partial learning is occuring, no learning is happening.

However, when this last layer is not initalised to zero (just multiplied by 0.1), then we say that the last layer does learn slightly and has non-zero gradients. However, these do not backpropagate at all to the deeper layers. We also note that the parameters of the last layer learn the fastest.

Next time to give a better answer straightaway:

1. Check the wieghts and the biases
2. Compare the loss with the loss if the model gave a uniform probability
