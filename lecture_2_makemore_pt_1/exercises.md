# Karpathy Makemore Pt1 Exercises

> E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?

The loss of the trigram model was 2.43. This is an improvement of 0.05 compared to the 2.48 loss of the bigram model in the Karpathy notebook.

> E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?

We see that the trigram model with the same L2 regularisation has a lower Negative Log-Likelihood (NLL) loss than the bigram model on both the dev and test datasets. We also note that the loss on the dev set is lower than the loss on the train set which is because the regularisation is only on the train set.

> E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?

The trigram model has the lowest loss on dev when the strength of smoothing is zero. This is maybe because the model is small relative to the dataset so it doesn't have the capacity to overfit. The loss increased as the strength of smoothing increased. At extremely high smoothing strength, it led to a gradient explosion where the loss is extremely high.

The final best loss achieved was 2.3764.

> E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?

Yes, we can. I did this for the train_bigram function in eval_bigram_vs_trigram.py.

> E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?

F.cross_entropy improves both the forward and backward pass:

1. Forward pass: F.cross_entropy prevents Infinity/NaNs by substracting from all logits the largest logits.
2. Backward pass: As F.cross_entropy combines the softmax and negative log-likelihood calculations, when doing backprop, the function can treat the two operations as a single fused node. The maths is below showing why how it simplifies. This means less memory is used, backprop is faster, and there are fewer floating point errors.

#### Gradient backwards propogation maths

##### Operations

1. Start with a raw logit for a correct: $z_c$

2. Convert to a probability using Softmax:

$$p_c = \frac{e^{z_c}}{\sum e^{z_j}}$$

1. NLL loss: $L = -log(p_c)$

##### Combine

1. $$L = -log(\frac{e^{z_c}}{\Sigma e^{z_j}})$$
2. $$L = -log(e^{z_c}) + log({\Sigma e^{z_j}})$$
3. $$L = -z_c + log({\Sigma e^{z_j}})$$

##### Take derivative

1. $$\frac{\partial L}{\partial z_c} = -1 + \frac{e^{z_c}}{\Sigma e^{z_j}}$$
2. This is the same as the original probability minus one, which means that the intermediate steps of finding the derivative for each operation can be skipped.
3. For intuition, imagine that the model gives the probability of the correct character to be 10%. Then the gradient of the logit node for the correct character given the previous character will be -0.9. This means that when the learning step happens, the weight will move up by 0.9 $\times$ x $\times$ learning_rate.
