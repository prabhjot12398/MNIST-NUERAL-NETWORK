# MNIST-NUERAL-NETWORK
Two-layer neural network and trained it on the MNIST digit recognizer dataset.
a[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image. A hidden layer  a[1]
will have 10 units with ReLU activation, and finally our output layer  a[2] will have 10 units corresponding
to the ten digit classes with softmax activation.
there are three parts to solve this 
1.forward propagation- Z^[1]=W^[1]X+b^[1]
                       A^[1]=gReLU(Z^[1]))     (Activation Fuction)
                      Z^[2]=W^[2]A^[1]+b^[2]
                      A^[2]=gsoftmax(Z^[2])    (Gives probabilities as each of the 10 nodes in the output can be 10 digits)

2.Backward propagation -  Here we are going to start with our prediction and find out how much the prediction actually deviated by the actual label
dZ^[2]=A^[2]−Y
 
dW^[2]=1/mdZ^[2]A^[1]T                         (dW^2 - derivative of the loss function with respect of the weights in layer 2)
dB^[2]=1/mΣdZ^[2]                              (db^2 - average of absolute error i.e how much the output is off)
dZ^[1]=W^[2]TdZ^[2].∗g^[1]′(z^[1])             (dZ^2 - how much the hidden layer is off where g is the derivate of the activation function because we need to undo the activation function to get proper error 
dW^[1]=1/mdZ^[1]A^[0]T
dB^[1]=1/mΣdZ^[1]

3.Parameter updates:

W[2]:=W[2]−αdW[2]                             (α - activation rate)
b[2]:=b[2]−αdb[2]
W[1]:=W[1]−αdW[1]
b[1]:=b[1]−αdb[1]
