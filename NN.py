"""
Simple Neural Network class
This site was useful: https://brilliant.org/wiki/backpropagation/
"""
import numpy as np

class nn:

    ## Activation functions and their derivatives    
    def __sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))
    __activation_lookup = {'sigmoid':__sigmoid}
        
    def __init__(self, nnodes, activation='sigmoid'):
        """
        inputs:
            nnodes - 1d array-like containing the number of nodes in each layer
                     including the input and output layer
            
            activation - string indicating which activation function to use.
                         Currently only sigmoid is supported. Will add support
                         for passing user-defined function.
        """
        self.nnodes = nnodes
        self.nlayers = len(nnodes)
        self.weights = [np.ones((nnodes[i+1],nnodes[i]+1)) for i in range(self.nlayers-1)]
        self.activation = self.__activation_lookup[activation]
    
    def randomize(self):
        """Randomizes the weights. Good thing to do before you start training"""
        self.weights = [np.random.random(np.shape(w)) for w in self.weights]
        
    def __forward(self, input, extended_output=False):
        """
        Do the forward propagation with current weights.
        Optionally return the outputs for each layer.
        """
        if extended_output:
            outputs = [np.zeros(n+1) for n in self.nnodes[:-1]]
        thislayer = np.copy(input)
        for i in range(self.nlayers-2):
            ## Add bias, mulitply by weights, activate
            thislayer = self.activation(self.weights[i] @ np.append(thislayer,1))
            if extended_output:
                outputs[i] = thislayer
        ## Last step without activation for output layer
        thislayer = self.weights[i+1] @ np.append(thislayer,1)
        if extended_output:
            outputs[i+1] = thislayer
            return outputs
        else:
            return thislayer
        
    def evaluate(self, input):
        """Evaluate the output given an input"""
        return self.__forward(input)
        
    def __backprop(self, input, expected):
        """Back propagate the error for a single input-output pair"""
        outputs = self.__forward(input, extended_output=True)
        dEda = 0.*np.copy(outputs)      ## partial derivative of the error wrt the activation
        dEdw = 0.*np.copy(self.weights) ## partial derivative of the error wrt the weight
        outputs = [input]+outputs ## include input layer
        ## Back propagate errors on each output, assumes sigmoid activation function
        for i in range(1,self.nlayers):
            if i == 1:
                dEda[-i] = expected - outputs[-1] ## Error on the final result
            else:
                dEda[-i] = outputs[-i]*(1-outputs[-i])*np.sum(dEda[-i+1] @ self.weights[-i+1])
            dEdw[-i] = np.outer(dEda[-i], np.append(outputs[-i-1],1))
            
        return dEdw
            
    def train(self, inputs, outputs, learning_rate=0.1):
        """Train the network given a set of input/expected output pairs."""
        ## Loop over io pairs. Can be multithreaded.
        dEdw = 0.*np.copy(self.weights)
        for i in range(len(inputs)):
            dEdw += self.__backprop(inputs[i], outputs[i])
        
        ## Update weights with average of derivatives
        self.weights += learning_rate*dEdw/len(inputs)
        
        
        
        
        
        
        
        
        
        
        
        
        