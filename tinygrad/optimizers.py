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



class RMSProp(Optimizer):

    def __init__(self, params, lr, gamma=0.9, eps=1e-6, weight_decay=None):
        """
        Implements RMSProp Optimization algorithm.

        Parameters:
        - params: parameter list of the model
        
        - lr: learning rate
        
        - gamma: RMSProp parameter

        - eps: constant to improve numerical stability

        - weight_decay: parameter for L2 regularization

        Returns:
        -   None. Performs weights update with step function.

        """
        if params is None or not isinstance(params, list):
            raise ValueError("params parameter should be of type <list>. Got ", type(params))
        if lr <= 0.0:
            raise ValueError("learning rate should be > 0.0. Got ", lr)
        if gamma <= 0.0 :
            raise ValueError("gamma should be > 0.0. Got ", gamma)
        if weight_decay is not None and weight_decay <= 0.0 :
             raise ValueError("weight_decay should be > 0.0. Got ", weight_decay)
        if eps is not None and eps <= 0.0 :
             raise ValueError("eps should be > 0.0. Got ", weight_decay)

        super().__init__(params)

        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.s_t = []
        for i in range(len(params)):
            self.s_t += [0]
        self.weight_decay = weight_decay

    def step(self):
        """
        Perform optimizations step
        """
        for idx, p in enumerate(self.params):

            self.s_t[idx] = self.gamma*self.s_t[idx] + (1.0-self.gamma)*p._grad**2
            if self.weight_decay is not None:
                p._grad += self.weight_decay * p.data
            
            p.data -= self.lr * (p._grad / (self.s_t[idx] + self.eps)**.5)

            
