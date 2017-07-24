import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    # shapes
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    for i in range(N):
        for j in range(C):
            A =  np.sum(np.exp(W.T.dot(X[i,:])))
            B =  np.exp(W[:,j].T.dot(X[i,:]))


            loss += (y[i]==j)*np.log(B/A)

    loss = -(1/float(N))*loss + 0.5*reg*(np.sum(W*W))
    for i in range(C):
        for j in range(N):
            A = np.exp(W[:,i].T.dot(X[j,:]))
            B = 1/np.sum(np.exp(W.T.dot(X[j,:])))
            dW[:,i] =  dW[:,i] - X[j,:].T*(1*(y[j]==i)-A*B)

        dW[:,i] =(1/float(N))*dW[:,i] + reg*W[:,i]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    L = np.zeros((N,C))
    A = np.sum(np.exp(X.dot(W)),axis=1)
    B = np.exp(X.dot(W))
    H = B/A[:,None]
    K = np.array(range(0,C))
    y.resize(y.shape[0],1)
    L = (y == K).astype(int)
    loss =  -(1.0/float(N))*np.sum(L*np.log(H)) + 0.5*reg*(np.sum(W*W))


    dW = -(1/float(N))*X.T.dot(L-H) +  reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

