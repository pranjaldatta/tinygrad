"""
currently optimizers are defined as simple functions (unlike PyTorch's 
implementation) because currently things like model saving/loading, state_dicts
are not supported
"""
class Optimizer(object):
    """
    Base optimizer class.
    
    Currently optimizers are defined as simple functions (unlike PyTorch's 
    implementation) because currently things like model saving/loading, state_dicts
    are not supported
    """
    def __init__(self, params):
        """
        Base optimizer class constructor.

        Parameters:
        - params: An iterable list of model parameters. Currently ONLY parameter
                  lists are supported.

        """
        self.params = params


class SimpleSGD(Optimizer):

    def __init__(self, params, lr, weight_decay=1e-5):
        """
        A Simple Implementation of the Stochastic Gradient Descent Optimization
        technique.

        Parameters:
        - params : An iterable list of model parameters. Currently ONLY parameter
                   lists are supported.

        -lr : learning rate for step optimization 
        
        - weight_decay : alpha parameter of L2 regularization loss. 

        Returns: 
            None. Performs weight update with step function. 
        
        """
        if params is None or not isinstance(params, list):
            raise ValueError("model parameters of invalid type. Required <list>, Found: ",type(params))
        if lr is None or lr < 0.0 :
            raise ValueError("Invalid learning rate. Learning rate allowed: >0.0. Found: ", lr)
        
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
    
    def step(self):
        """
        Perform optimization step
        """

        for p in self.params:
            
            if self.weight_decay is not None: #L2 Regularization
                p._grad += self.weight_decay * p.data 
            
            p.data -= self.lr * p._grad


