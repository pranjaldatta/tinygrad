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


class Adam(Optimizer):

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=None):
        """
        Implements Adam Optimization Algorithm.

        Parameters:
        - params: parameter list of the model

        - lr: initial learning rate

        - beta1: coefficient of running average of gradient (similiar to momentum)

        - beta2: coefficient of running average of squares of gradients
                 (similar to RMSProp)

        - eps: constant to improve numerical stability

        - weight_decay: L2 regularization parameter

        Returns:
        -   None. Updates weights with step function
        """
        if params is None or not isinstance(params, list):
            raise ValueError("params parameter should be of type <list>. Got ", type(params))
        if lr <= 0.0:
            raise ValueError("learning rate should be > 0.0. Got ", lr)
        if beta1 <= 0.0 :
            raise ValueError("beta1 should be > 0.0. Got ", beta1)
        if beta2 <= 0.0 :
            raise ValueError("beta2 should be > 0.0. Got ", beta2)
        if weight_decay is not None and weight_decay <= 0.0 :
             raise ValueError("weight_decay should be > 0.0. Got ", weight_decay)
        if eps is not None and eps <= 0.0 :
             raise ValueError("eps should be > 0.0. Got ", weight_decay)
        
        super().__init__(params)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m_t = []
        self.v_t = []
        for _ in range(len(self.params)):
            self.m_t += [0]
            self.v_t += [0]

    def step(self):
        """
        Performs the optimization step
        """
        for idx, p in enumerate(self.params):

            # calculating initial running average for each param
            self.m_t[idx] = self.beta1*self.m_t[idx] + (1.0-self.beta1)*p._grad
            self.v_t[idx] = self.beta2*self.v_t[idx] + (1.0-self.beta2)*p._grad**2

            # bias-correcting them as suggested in the paper
            m_cap = self.m_t[idx]/(1.0 - self.beta1**(idx+1))
            v_cap = self.v_t[idx]/(1.0 - self.beta2**(idx+1))

            if self.weight_decay is not None:
                p._grad += self.weight_decay * p.data
            
            p.data -= self.lr * (m_cap/(v_cap**0.5 + self.eps))




            
