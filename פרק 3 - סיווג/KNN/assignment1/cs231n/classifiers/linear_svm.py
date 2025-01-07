from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs:
    - W: Weights (D, C)
    - X: Data (N, D)
    - y: Labels (N,)
    - reg: Regularization strength

    Returns:
    - loss: Scalar loss value
    - dW: Gradient of loss with respect to W
    """
    # Initialize the gradient as zero
    dW = np.zeros(W.shape)

    # Number of training samples
    num_train = X.shape[0]

    # Step 1: Compute scores
    scores = X.dot(W)  # Shape: (N, C)

    # Step 2: Compute correct class scores
    correct_class_scores = scores[np.arange(num_train), y]  # Shape: (N,)

    # Step 3: Compute margins
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)  # Shape: (N, C)
    margins[np.arange(num_train), y] = 0  # Correct class margin should be zero

    # Step 4: Compute loss
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)  # Add regularization

    # Step 5: Compute gradient
    # Create mask for positive margins
    positive_margins_mask = (margins > 0).astype(float)  # Shape: (N, C)

    # Subtract counts for correct class
    row_sum = np.sum(positive_margins_mask, axis=1)  # Shape: (N,)
    positive_margins_mask[np.arange(num_train), y] -= row_sum

    # Compute dW
    dW = X.T.dot(positive_margins_mask) / num_train  # Shape: (D, C)

    # Add regularization gradient
    dW += 2 * reg * W

    return loss, dW
