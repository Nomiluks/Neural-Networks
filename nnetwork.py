import numpy as np
from layers import *

#####################################################################
# TODO: Implement init_three_layer_neuralnet and 
# three_layer_neuralnetwork functions
#####################################################################


def init_three_layer_neuralnet(weight_scale=1, bias_scale=0, input_feat_dim=786,
                           num_classes=10, num_neurons=(20, 30)):
  """
  Initialize the weights for a three-layer NeurAlnet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_feat_dim: number of features of input examples..
  - num_classes: The number of classes for this network. Default is 10   (for MNIST)
  - num_neurons: A tuple containing number of neurons in each layer...
  
  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1 (input_feat_dim,num_neurons[0]), b1: Weights and biases for the affine layer
    - W2, b2: Weights and biases for the affine layer
    - W3, b3: Weights and biases for the affine layer    
  """
  
  assert len(num_neurons)  == 2, 'You must provide number of neurons for two layers...'

  input_size = input_feat_dim
  H1 = num_neurons[0]
  H2 = num_neurons[1]
  outputLayer = num_classes

  #w = np.random.randn(n) * sqrt(2.0/n)
  model = {}
  model['W1'] = weight_scale*np.random.randn(input_size, H1)*np.sqrt(2.0/input_size) # Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['b1'] = bias_scale*np.zeros((H1)) # Initialize with zeros
  
  model['W2'] = weight_scale*np.random.randn(H1, H2)*np.sqrt(2.0/H1) # Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['b2'] = bias_scale*np.zeros((H2))# Initialize with zeros

  model['W3'] = weight_scale*np.random.randn(H2, outputLayer)*np.sqrt(2.0/H2) # Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['b3'] = bias_scale*np.zeros((outputLayer))# Initialize with zeros

  return model



def three_layer_neuralnetwork(X, model, y=None, reg=0.0,verbose=0):
  """
  Compute the loss and gradient for a simple three-layer NeurAlnet. The architecture
  is affine-relu-affine-relu-affine-softmax. We use L2 regularization on 
  for the affine layer weights.

  Inputs:
  - X: Input data, of shape (N,D), N examples of D dimensions
  - model: Dictionary mapping parameter names to parameters. A three-layer Neuralnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the affine layer
    - W2, b2: Weights and biases for the affine layer
    - W3, b3: Weights and biases for the affine layer
    
  - y: Vector of labels of shape (D,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3']
  dW1,dW2,dW3,db1,db2,db3=np.zeros_like(W1),np.zeros_like(W2),np.zeros_like(W3),np.zeros_like(b1),np.zeros_like(b2),np.zeros_like(b3)
  
  N,D = X.shape
  C = b3.shape[0]
  assert W1.shape[0] == D, ' W1 2nd dimenions must match number of features'

  scores = np.zeros((N, C)) #declaring score variable to store scores
  #print scores.shape
  
  #'''
  #################################### written by me! ###########################################
  # Compute the forward pass
    
  #layer 1
  a1, cache_a1         = affine_forward(X,W1,b1)
  a1_out, cache_relu1  = relu_forward(a1)
    
  #layer 2
  a2, cache_a2         = affine_forward(a1_out,W2,b2)
  a2_out, cache_relu2  = relu_forward(a2)

  #layer 3
  scores, cache_a3     = affine_forward(a2_out,W3,b3)

  
  #################################################################################################
  #'''
  
  if verbose:
    print ['Layer {} Variance = {}'.format(i+1, np.var(l[:])) for i,l in enumerate([a1, a2, cache_a3[0]])][:]
  if y is None:
    return scores

  ########### compute the gradients ###########
  #softmax layer and dout
  data_loss, dout  = softmax_loss(scores, y)  

  #http://cs231n.github.io/neural-networks-case-study/
  reg_loss = (0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)+0.5 * reg * np.sum(W3 * W3))


  #'''
  #################################### written by me! ###########################################
  # Compute the backward pass
  #layer 3 
  dout, dW3, db3 = affine_backward(dout, cache_a3)
  dW3 += reg * W3

  #layer 2
  dout = relu_backward(dout, cache_relu2)
  dout, dW2, db2 = affine_backward(dout, cache_a2)
  dW2 += reg * W2
    
  #layer 1
  dout = relu_backward(dout, cache_relu1)
  dout, dW1, db1 = affine_backward(dout, cache_a1)  
  dW1 += reg * W1 
  ##################################################################################################
  #'''
    
  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,'W3':dW3,'b3':db3}
  
  return loss, grads

