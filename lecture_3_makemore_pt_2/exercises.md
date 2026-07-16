# Karpathy Makemore Pt2 Exercises

> E01: Tune the hyperparameters of the training to beat my best validation loss of 2.2

## Hyperparameters

* Number of neurons in the hidden layer

* Dimensionality of embedding lookup table

* Number of characters fed in as context to the model

* Number of learning iterations

* Learning rate - what is the best decay?

* Batch size - improve convergence speed?

* More ideas in the Bengio paper

## Logs

1. Base copy of Karpathy's code, 200 iterations. Loss: **2.2923**

2. Same setup with 200,000 iterations. Loss: **2.1376**

3. Increased context length from 3 to 4. Loss: **2.1930** (Note: Loss is up)

4. Reverted block size back to 3. Loss: **2.1687**

5. Increased batch size from 32 to 48. Loss: **2.1413**

6. Increased batch size from 48 to 64. Test Loss: **2.1294** (Note: Showing signs of overfitting, as the training loss is now 1.66 and the dev set is 2.0424)

7. Increased number of dimensions to 20. The idea is that more information to learn will reduce overfitting. Test Loss: **2.1352** (Note: No improvement)

8. Reverted dimensions back to 10.

9. Increased size of the hidden layer from 200 to 300. Test Loss: **2.1605**

10. Kept the hidden layer at 300 with dimensions = 15. Loss: **2.1158** (Note: Best yet)

11. Changed to uniformly distributed weights on the output layer. Loss: **2.1049**

12. Tried Kaiming initialization on the hidden layer. Loss: **2.0851**

13. Added another hidden layer with 300 neurons (rectangular neural network). Test/Dev Loss: 11.7108 and training loss: 2.02 and time to run: 2minutes. This seems to be huge overfitting, why?

14. Potentially due to using fan_in from the first input layer to do Kaiming initalisation on second hidden layer. Changed loss: 8.4788, still really bad

15. Realised that the functions doing the forward pass on the dev and test sets were incorrect, which was causing the high loss. Corrected, the loss is 2.0866!

> E02: I was not careful with the initialization of the network in this video. (1) What is the loss you'd get if the predicted probabilities at initialization were perfectly uniform? What loss do we achieve? (2) Can you tune the initialization to get a starting loss that is much more similar to (1)?

When perfectly uniform, each character has a 1/27 probability of being chosen, which gives an average Negative Log-Likelihood (NLL) of -log(1/27) = 3.2958.

We can tune the initialization to be more similar to the uniform distribution by making each weight very close to zero. We do this by multiplying the weights matrix by 0.01 and setting the bias matrix to zero. Then, when the logits are exponentiated, they are all roughly 1, which after normalization means their probability is 1/27.

> E03: Read the Bengio et al 2003 paper (link above), implement and try any idea from the paper. Did it work?

Idea to try from the paper: Include more hidden layers. This isn't explicitly in the paper, but a lot of the text seemed partly focused on synthesizing advanced n-gram models with the neural network, or having a direct link from the input to the output to skip the hidden layer.
