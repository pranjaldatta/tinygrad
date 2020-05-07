import random
from .core import Tensor

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p._grad = 0

    def parameters(self):
        return []  
    

        


class Neuron(Module):
    """
    since this basic module only supports scalar values
    we split up a hidden layer into its individual neuorons
    and perform operations neuron wise.
    """
    def __init__(self, num_in, nonlin=True):
        self.weights = [Tensor(random.uniform(-1.0, 1.0)) for _ in range(num_in)]
        #self.weights = [Tensor(2) for _ in range(num_in)]
        self.bias = Tensor(0)
        self.nonlin = nonlin

    def __call__(self, x):
            result = []
            for wi, xi in zip(self.weights, x):
                result += [wi*xi]
            result = sum(result, self.bias)
            if self.nonlin:
                return result.relu()
            return result 

    def parameters(self):
        return self.weights + [self.bias]  #last element in params is the bias

    def __repr__(self):
        return "Neuron(weights:{}, relu:{})".format(len(self.weights), self.nonlin)


class Layer(Module):

    def __init__(self, ins, outs, nonlin):
        """
        defines a layer.
        
        params:
        -> ins: number of incoming connections for each neuron
        -> outs: number of outgoing connections (also number of neurons in the
           given layer)   
        """
        self.ins = ins
        self.outs = outs
        self.nonlin = nonlin
        self.neurons = [Neuron(self.ins, nonlin=self.nonlin) for _ in range(outs)]
        #self.flatten = lambda l:[item for sublist in l for item in sublist]

    def __call__(self, x):
        """
        runs input variable x through a given layer

        params: 
        -> x: input variable of type Tensor
        """
        result = [neuron(x) for neuron in self.neurons]
        if len(result) == 1:
            result = result[0]
        return result

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
    def __repr__(self):
        return "Linear(ins:{} outs:{} num_parameters:{})".format(self.ins,\
                self.outs, len(self.parameters())
                )

class SimpleMLP(Module):
    """
    defines a simple MLP and runs an input through it 

    parameters: 
    -> num_in: number of input nodes (i.e. input layer)
    -> num_out: number of output layer (i.e. output layer)
    -> hidden: a list of hidden layer sizes. default: []
    -> nonlin: non linearity applied to all nodes. defualt: True
    """
    def __init__(self, num_in, num_out, hidden=[], nonlin=True):
        self.num_in = num_in
        self.num_out = num_out
        self.hidden = hidden
        self.nonlin = nonlin
        self.hidden = [self.num_in] + self.hidden + [self.num_out]
        self.layers = [Layer(self.hidden[i], self.hidden[i+1], self.nonlin) for i in range(len(self.hidden)-1)]
    
    def __call__(self, x):
        """
        pass an input x through the layers
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"SimpleMLP({','.join(str(layer) for layer in self.layers)})"
    
    def summary(self):
        print("SimpleMLP(")
        for layer in self.layers:
            print(layer)
        print(")")
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]




