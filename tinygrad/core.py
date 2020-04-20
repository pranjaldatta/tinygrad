class Tensor:
    """
    The base Tensor class. Wraps around a data item and stores 
    addition variables that help in backward diff
    """
    def __init__(self, data, _parents=[], _operator=''):
       
        self.data = data
        self._parents = _parents
        self._operator = _operator
        self._grad = 0
        self._backward = lambda: None
        
    def __repr__(self):
        return "Tensor({}, _grad:{})".format(self.data, self._grad)
    
    def __add__(self, var):

        var = var if isinstance(var, Tensor) else Tensor(var)
        result = Tensor(self.data + var.data, [self, var], '+')

        def _backward():
            self._grad += result._grad
            var._grad += result._grad
        result._backward = _backward
        
        return result

    def __mul__(self, var):

        var = var if isinstance(var, Tensor) else Tensor(var)
        result = Tensor(self.data*var.data, [self, var], '*')

        def _backward():
            self._grad += var.data*result._grad
            var._grad += self.data*result._grad
        result._backward = _backward    
        
        return result

    def __pow__(self, var):

        result = Tensor(self.data**var, [self,], '**')

        def _backward():
            self._grad += (var*self.data**(var-1))*result._grad
        result._backward = _backward
        
        return result

    def backward(self):
        """
        builds the topological order of the tensors as executed
        and runs _backward on each tensor
        """

        visited = set()
        topo_order = []
        def create_order(t):
            if t not in visited:
                visited.add(t)
                for parent in t._parents:
                    create_order(parent)
                topo_order.append(t)
        create_order(self)
        
        self._grad = 1
        for t in reversed(topo_order):
            t._backward()



        

