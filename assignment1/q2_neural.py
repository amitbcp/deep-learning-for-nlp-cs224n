 #!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation

    #This is for the Input of the hidden layer H
    # h = sigmoid(np.dot(data,W1)+b1)
    z1 = np.dot(data,W1) + b1
    a1 = sigmoid(z1)

    #This is the output Layer
    z2 = np.dot(a1,W2) +b2
    y_pred = softmax(z2)

    cost = - np.sum(labels * np.log(y_pred))

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    d3 = y_pred - labels
    gradW2 = np.dot(a1.T,d3)
    gradb2 = np.sum(d3,axis=0)

    da1 = np.dot(d3,W2.T)
    dz1 = sigmoid_grad(a1)*da1

    gradW1 = np.dot(data.T,dz1)
    gradb1 = np.sum(dz1,axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20 #Number of Training examples
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    #print("Data :: ",data.shape)
    labels = np.zeros((N, dimensions[2]))
    print("Labels Shape:: ",labels.shape)
    for i in list(range(N)):
        #Making it a one-hot vector from All Zeros
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    #print("Labels Value :: ",labels)

    ## Unfolding All the Parameters
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    print("Param Shape:: " , params.shape)
    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
