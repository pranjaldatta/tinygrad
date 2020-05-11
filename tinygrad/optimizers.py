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


class Adagrad(Optimizer):
    
    def __init__(self, params, lr=0.01, eps=1e-8, weight_decay=None):
        """
        Implements Adagrad Optimization Algorithm.

        Parameters:
        - params: parameter list of the model

        - lr: initial learning rate

        - eps: constant to improve numerical stability

        - weight_decay: L2 regularization parameter

        Returns:
        -   None. Updates weights with step function
        """

        if params is None or not isinstance(params, list):
            raise ValueError("params parameter should be of type <list>. Got ", type(params))
        if lr <= 0.0:
            raise ValueError("learning rate should be > 0.0. Got ", lr)
        if weight_decay is not None and weight_decay <= 0.0 :
             raise ValueError("weight_decay should be > 0.0. Got ", weight_decay)
        if eps is not None and eps <= 0.0 :
             raise ValueError("eps should be > 0.0. Got ", weight_decay)
        
        super().__init__(params)
        
        self.lr = lr 
        self.eps = eps
        self.weight_decay = weight_decay
        self.g_t = []
        for _ in range(len(self.params)):
            self.g_t += [0]
        
    def step(self):
        """
        Performs the optimization step
        """
        for idx, p in enumerate(self.params):

            # storing the running squares of the gradients
            self.g_t[idx] += p._grad**2

            if self.weight_decay is not None:
                p._grad += self.weight_decay * p.data
            
            p.data -= self.lr * (p._grad / (self.g_t[idx]+self.eps)**0.5)


class Adadelta(Optimizer):
    
    def __init__(self, params, rho=0.9, lr=1.0, eps=1e-6, weight_decay=None):
        """
        Implements Adadelta Optimization Algorithm.

        Parameters:
        - params: parameter list of the model

        - rho: coefficient used for computing a running average of 
               squared gradients 

        - lr: initial learning rate

        - eps: constant to improve numerical stability

        - weight_decay: L2 regularization parameter

        Returns:
        -   None. Updates weights with step function
        """

        if params is None or not isinstance(params, list):
            raise ValueError("params parameter should be of type <list>. Got ", type(params))
        if lr <= 0.0:
            raise ValueError("learning rate should be > 0.0. Got ", lr)
        if weight_decay is not None and weight_decay <= 0.0 :
             raise ValueError("weight_decay should be > 0.0. Got ", weight_decay)
        if eps is not None and eps <= 0.0 :
             raise ValueError("eps should be > 0.0. Got ", weight_decay)
        if rho is not None and rho <= 0.0:
            raise ValueError("rho should be > 0.0. Got ", rho)


        super().__init__(params)

        self.rho = rho
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps

        self.g_t = []
        self.delta_t = []
        for _ in range(len(self.params)):
            self.g_t += [0]
            self.delta_t += [0]


    def step(self):
        """
        Performs the optimization step
        """
        for idx, p in enumerate(self.params):
            
            self.g_t[idx] = self.rho * self.g_t[idx] + (1.0 - self.rho) * p._grad**2

            if self.weight_decay is not None:
                p._grad += self.weight_decay * p.data
            
            _g = ((self.delta_t[idx]+self.eps)**0.5)/((self.g_t[idx] + self.eps)**0.5)
            _g *= p._grad

            p.data -= _g   
        
            self.delta_t[idx] = self.rho * self.delta_t[idx] + (1.0 - self.rho) * _g**2


class Adamax(Optimizer):

    def __init__(self, params, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=None):
        """
        Implements the Adamax Optimizer.

        Parameters:
        - params: parameter list of the model

        - lr: initial learning rate of the model

        - beta1: coefficient to calculate running average of gradients

        - beta2: coefficient to calculate running average of square of
                 gradients.

        - eps: constant to improve numerical stability

        - weight_decay: L2 penalty parameters

        Returns:
        -   None. Updates weights with step function.
        """
        if params is None or not isinstance(params, list):
            raise ValueError("params parameter should be of type <list>. Got ", type(params))
        if lr <= 0.0:
            raise ValueError("learning rate should be > 0.0. Got ", lr)
        if weight_decay is not None and weight_decay <= 0.0 :
             raise ValueError("weight_decay should be > 0.0. Got ", weight_decay)
        if eps is not None and eps <= 0.0 :
             raise ValueError("eps should be > 0.0. Got ", weight_decay)
        if beta1 is not None and beta1 <= 0.0:
            raise ValueError("rho should be > 0.0. Got ", rho)
        if beta2 is not None and beta2 <= 0.0:
            raise ValueError("beta2 should be > 0.0. got ", beta2)

        super().__init__(params)

        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2

        self.m_t = []
        self.u_t = []
        for _ in range(len(self.params)):
            self.m_t += [0]
            self.u_t += [0]
        
    def step(self):
        """
        Performs the optimization step.
        """
        for idx, p in enumerate(self.params):

            self.m_t[idx] = self.beta1*self.m_t[idx] + (1.0-self.beta1)*p._grad
            self.u_t[idx] = max(self.beta2*self.u_t[idx], abs(p._grad))

            # bias correcting running average of gradient
            m_cap = self.m_t[idx]/(1.0 - self.beta1**(idx+1))

            p.data -= self.lr * (m_cap / (self.u_t[idx] + self.eps))
