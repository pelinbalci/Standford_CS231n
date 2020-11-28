from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dim_size = x[0].shape
    # (N, D) x (D, M) + (1, M)
    out = x.reshape(x.shape[0], np.prod(dim_size)).dot(w) + b.reshape(1, -1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dim_size = x[0].shape
    X = x.reshape(x.shape[0], np.prod(dim_size))
    dw = X.T.dot(dout)  # xt D*N, dout N*M
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # The ReLU function is defined as: For x > 0 the output is x, i.e. f(x) = max(0,x)
    # So for the derivative f '(x) it's actually:
    # if x < 0, output is 0. if x > 0, output is 1.

    dx = dout * (x > 0)
    # dx = dout
    # dx[x<=0] = 0
    # dx[x>0] = 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_mean = np.mean(x,axis=0)
        x_var = np.var(x, axis=0)
        x_std = np.sqrt(x_var +eps)
        x_norm = (x - x_mean) / x_std
        out = gamma * x_norm + beta
        cache = (gamma, x_norm, x_mean, x_var, x_std, x, eps)

        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Ref: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    gamma, x_norm, x_mean, x_var, x_std, x, eps = cache
    N, D = x_norm.shape
    dbeta = np.sum(dout, axis=0)
    dgammax = dout
    dgamma = np.sum(dout * x_norm, axis=0)
    dxnorm = dgammax * gamma
    dx_mean_1 = dxnorm * 1/x_std
    dinverse_var = np.sum(dxnorm * (x-x_mean), axis=0)

    dsqrt = (-1) / (np.sqrt(x_var + eps) ** 2) * dinverse_var
    dmean_1 = 0.5 * 1 / np.sqrt(x_var + eps) * dsqrt

    dsquare = 1. /N * np.ones((N,D)) * dmean_1
    dx_mean_2 =2 * (x-x_mean) * dsquare
    dmean_2 = -1 * np.sum(dx_mean_1 + dx_mean_2, axis=0)
    dx_1 = dx_mean_1 + dx_mean_2
    dx_2 = 1./N * np.ones((N,D)) * dmean_2
    dx = dx_1 + dx_2


    # Other way
    # dgamma = np.sum(dout * x_norm, axis=0)
    # dbeta = np.sum(dout, axis=0)
    #
    # dy_xnorm = gamma * dout
    # dxnorm_x_mu = 1 / x_std * dy_xnorm
    # dxnorm_std = - np.sum((x - x_mean) * dy_xnorm, axis=0) / (x_std ** 2)  # check
    # dvar_std = 1 / 2 * ((x_var + eps) ** (-1 / 2)) * dxnorm_std  # check
    # dmean_x = 1 / N * dvar_std  # check
    # dxnorm_x_mu += 2 * (x - x_mean) * dmean_x  # dvar /dx
    # dmean = np.sum(dxnorm_x_mu, axis=0) / N
    #
    # dx = dxnorm_x_mu - dmean



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    #
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma, x_norm, x_mean, x_var, x_std, x, eps = cache
    N, D = x_norm.shape

    dx_norm = dout * gamma
    dx_var = np.sum(dx_norm * (x - x_mean) * (-0.5) * (x_std ** -3), axis=0)
    dx_mean = np.sum(dx_norm * (-1.) / x_std, axis=0) + dx_var * np.sum(-2 * (x - x_mean)) / N
    dx = dx_norm * (1. / x_std) + dx_var * 2 * (x - x_mean) / N + dx_mean / N
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)


    # Ref: https://github.com/williamchan/cs231-assignment2/blob/master/cs231n/layers.py

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #
    # x_mean = np.mean(x, axis=1, keepdims=True)  # 4 values. shape:4
    # x_var = np.var(x, axis=1, keepdims=True)
    # x_std = np.sqrt(x_var + eps)
    # x_norm = ((x - x_mean) / x_std)
    # out = gamma * x_norm + beta
    # cache = (gamma, x_norm, x_mean, x_var, x_std, x, eps)
    #

    # Ref: https://github.com/israfelsr/CS231n/blob/master/assignment2/cs231n/layers.py
    mu = np.mean(x, axis=1, keepdims=True)  # (1 / N) * np.sum(x, axis=0) N,1
    var = np.var(x, axis=1, keepdims=True)  # (1 / N) * np.sum(((x - mu) ** 2), axis=0) N,1
    var_inv = 1 / np.sqrt(var + eps)  # N,1
    x_mu = x - mu  # N,D
    x_norm = x_mu * var_inv  # N,D
    out = gamma * x_norm + beta  # N,D
    cache = (gamma, x_norm, x_mu, var_inv)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # gamma, x_norm, x_mean, x_var, x_std, x, eps = cache
    # N, D = x_norm.shape
    # dbeta = np.sum(dout, axis=0)
    # dgammax = dout
    # dgamma = np.sum(dout * x_norm, axis=0)
    # dxnorm = dgammax * gamma
    # dx_mean_1 = (dxnorm * 1 / x_std)
    # dinverse_var = np.sum(dxnorm * (x - x_mean), axis=1, keepdims=True)
    #
    # dsqrt = (-1) / (np.sqrt(x_var + eps) ** 2) * dinverse_var
    # dmean_1 = 0.5 * 1 / np.sqrt(x_var + eps) * dsqrt
    #
    # dsquare = 1. / N * np.ones((N, D)) * dmean_1
    # dx_mean_2 = 2 * (x - x_mean) * dsquare
    # dmean_2 = -1 * np.sum(dx_mean_1 + dx_mean_2, axis=1)
    # dx_1 = dx_mean_1 + dx_mean_2
    # dx_2 = 1. / N * np.ones((D, N)) * dmean_2
    # dx = dx_1 + dx_2.T

    # Other:
    gamma, x_norm, x_mu, var_inv = cache
    N, D = x_norm.shape
    dgamma = np.sum(dout * x_norm, axis=0)  # N,D -> D,
    dxnorm = dout * gamma  # N,D -> N,D
    dbeta = np.sum(dout, axis=0)  # N,D -> D,

    dxmu = dxnorm * var_inv  # N,D -> N,D
    dvar_inv = np.sum(dxnorm * x_mu, axis=1, keepdims=True)  # N,D -> N,

    dvar = dvar_inv * -0.5 * var_inv ** 3  # N,
    dx = dxmu

    dxmu += dvar * 2 / D * x_mu
    dmu = -1 * np.sum(dxmu, axis=1, keepdims=True)

    dx += 1 / D * dmu

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) <p) /p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    N, C, H, W = x.shape
    N, C, H_pad, W_pad= x_pad.shape
    F, C, HH, WW = w.shape

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    H_out= int(H_out)
    W_out = int(W_out)

    out = np.zeros((N, F, int(H_out),int(W_out)))

    for n in range(N):
        for f in range(F):
            for h_s in range(0, H_out):
                for w_s in range(0, W_out):
                    out[n, f, h_s, w_s] = (x_pad[n, :, stride * h_s: HH + stride * h_s, stride * w_s: WW + stride * w_s] * w[f, :, :, :]).sum() + b[f]
                    #for c in range(C):
                        #out[n][f][w_s][h_s] += np.sum(x_pad[n][c][:, 2*h_s : HH + 2*h_s][2*w_s : WW + 2*w_s] * w[f][c]) + b[f] / 3



    # Control:
    # out[n][f][0][0] += np.sum(x_pad[n][c][:, :HH][:WW] * w[f][c]) + b[f]/3
    # out[n][f][0][1] += np.sum(x_pad[n][c][:, 2:HH + 2][:WW] * w[f][c]) + b[f]/3
    # out[n][f][1][1] += np.sum(x_pad[n][c][:, 2:HH + 2][2:WW+2] * w[f][c]) + b[f]/3
    # out[n][f][1][0] += np.sum(x_pad[n][c][:, :HH][2:WW + 2] * w[f][c]) + b[f]/3
    # np.sum(x_pad[0][0][:, :4][:4] * w[0][0]) + np.sum(x_pad[0][1][:, :4][:4] * w[0][1])+ np.sum(x_pad[0][2][:, :4][:4] * w[0][2]) +b[0]
    # np.sum(x_pad[0][0][:, 2:6][:4] * w[0][0]) + np.sum(x_pad[0][1][:, 2:6][:4] * w[0][1]) + np.sum(x_pad[0][2][:, 2:6][:4] * w[0][2]) +b[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Ref: https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment2/cs231n/layers.py
    x, w, b, conv_param = cache

    stride = conv_param['stride']
    pad = conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    N, C, H, W = x.shape
    N, C, H_pad, W_pad= x_pad.shape
    F, C, HH, WW = w.shape

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    H_out= int(H_out)
    W_out = int(W_out)

    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            db[f] += dout[n, f].sum()
            for h_s in range(0, H_out):
                for w_s in range(0, W_out):
                    dx_pad[n, :, stride * h_s: HH + stride * h_s, stride * w_s: WW + stride * w_s] += w[f, :, :, :] * dout[n, f, h_s, w_s]
                    dw[f, :, :, :] += (x_pad[n, :, stride * h_s: HH + stride * h_s, stride * w_s: WW + stride * w_s]) * dout[n, f, h_s, w_s]

    # extract dx from dx_pad
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    p_h = pool_param['pool_height']
    p_w = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape

    H_out = 1 + (H - p_h) // stride
    W_out = 1 + (W - p_w) // stride

    out = np.zeros((N, C, int(H_out), int(W_out)))

    for n in range(N):
        for c in range(C):
            for h_s in range(0, H_out):
                for w_s in range(0, W_out):
                    out[n, c, h_s, w_s] = np.max(x[n, c, stride * h_s: p_h + stride * h_s, stride * w_s: p_w + stride * w_s])






    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    p_h = pool_param['pool_height']
    p_w = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape

    H_out = 1 + (H - p_h) // stride
    W_out = 1 + (W - p_w) // stride

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h_s in range(0, H_out):
                for w_s in range(0, W_out):
                    sub_x = x[n, c, stride * h_s: p_h + stride * h_s, stride * w_s: p_w + stride * w_s]
                    max_row_idx = np.argmax(np.max(sub_x, axis=1))
                    max_col_idx = np.argmax(np.max(sub_x, axis=0))
                    dx[n, c, stride * h_s: p_h + stride * h_s, stride * w_s: p_w + stride * w_s][max_row_idx][max_col_idx] += dout[n, c, h_s, w_s]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Ref:https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment2/cs231n/layers.py
    N, C, H, W = x.shape
    # transpose to a channel-last notation (N, H, W, C) and then reshape it to
    # norm over N*H*W for each C
    x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)

    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # transpose the output back to N, C, H, W
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Ref:https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment2/cs231n/layers.py
    N, C, H, W = dout.shape

    # transpose to a channel-last notation (N, H, W, C) and then reshape it to
    # norm over N*H*W for each C
    dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)

    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # transpose the output back to N, C, H, W
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Ref:https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment2/cs231n/layers.py

    # using minimal-num-of-operations-per-step policy to ease the backward pass

    N, C, H, W = x.shape
    size = (N * G, C // G * H * W)  # in groupnorm, D = C//G * H * W

    # (0) rehsape X to accommodate G
    # divide each sample into G groups (G new samples)
    x = x.reshape((N * G, -1))  # reshape to same as size # reshape NxCxHxW ==> N*GxC/GxHxW =N1*C1 (N1>N*Groups)

    # (1) mini-batch mean by averaging over a particular column / feature dimension (D)
    # over each sample (N) in a minibatch
    mean = x.mean(axis=1, keepdims=True)  # (N,1) # sum through D
    # can also do mean = 1./N * np.sum(x, axis = 1)

    # (2) subtract mean vector of every training example
    dev_from_mean = x - mean  # (N,D)

    # (3) following the lower branch for the denominator
    dev_from_mean_sq = dev_from_mean ** 2  # (N,D)

    # (4) mini-batch variance
    var = 1. / size[1] * np.sum(dev_from_mean_sq, axis=1, keepdims=True)  # (N,1)
    # can also do var = x.var(axis = 0)

    # (5) get std dev from variance, add eps for numerical stability
    stddev = np.sqrt(var + eps)  # (N,1)

    # (6) invert the above expression to make it the denominator
    inverted_stddev = 1. / stddev  # (N,1)

    # (7) apply normalization
    # note that this is an element-wise multiplication using broad-casting
    x_norm = dev_from_mean * inverted_stddev  # also called z or x_hat (N,D)
    x_norm = x_norm.reshape(N, C, H, W)

    # (8) apply scaling parameter gamma to x
    scaled_x = gamma * x_norm  # (N,D)

    # (9) shift x by beta
    out = scaled_x + beta  # (N,D)

    # backprop sum axis
    axis = (0, 2, 3)

    # cache values for backward pass
    cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, \
             'beta': beta, 'eps': eps, 'x_norm': x_norm, 'dev_from_mean': dev_from_mean, \
             'inverted_stddev': inverted_stddev, 'x': x, 'axis': axis, 'size': size, 'G': G, 'scaled_x': scaled_x}

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Ref:https://github.com/amanchadha/stanford-cs231n-assignments-2020/blob/master/assignment2/cs231n/layers.py

    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, axis, size, G, scaled_x = \
        cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
        cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
        cache['mean'], cache['axis'], cache['size'], cache['G'], cache['scaled_x']

    N, C, H, W = dout.shape

    # (9)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # 1xCx1x1
    dscaled_x = dout  # N1xC1xH1xW1

    # (8)
    dgamma = np.sum(dscaled_x * x_norm, axis=(0, 2, 3),
                    keepdims=True)  # N = sum_through_D,W,H([N1xC1xH1xW1]xN1xC1xH1xW1)
    dx_norm = dscaled_x * gamma  # N1xC1xH1xW1 = [N1xC1xH1xW1] x[1xC1x1x1]
    dx_norm = dx_norm.reshape(size)  # (N1*G,C1//G*H1*W1)

    # (7)
    dinverted_stddev = np.sum(dx_norm * dev_from_mean, axis=1, keepdims=True)  # N = sum_through_D([NxD].*[NxD]) =4×60
    ddev_from_mean = dx_norm * inverted_stddev  # [NxD] = [NxD] x [Nx1]

    # (6)
    dstddev = (-1 / (stddev ** 2)) * dinverted_stddev  # N = N x [N]

    # (5)
    dvar = 0.5 * (1 / np.sqrt(var + eps)) * dstddev  # N = [N+const]xN

    # (4)
    ddev_from_mean_sq = (1 / size[1]) * np.ones(size) * dvar  # NxD = NxD*N

    # (3)
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq  # [NxD] = [NxD]*[NxD]

    # (2)
    dx = (1) * ddev_from_mean  # [NxD] = [NxD]
    dmean = -1 * np.sum(ddev_from_mean, axis=1, keepdims=True)  # N = sum_through_D[NxD]

    # (1) cache
    dx += (1 / size[1]) * np.ones(size) * dmean  # NxD (N= N1*Groups) += [NxD]XN

    # (0):
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
