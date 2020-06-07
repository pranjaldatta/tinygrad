import math

class Tensor:
    """
    The base Tensor class. Wraps around a data item and stores 
    additional variables that help in backward diff
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

    def relu(self):

        result = 0 if self.data < 0 else self.data
        result = Tensor(result, (self,), "ReLU")

        def _backward():
            self._grad += (result.data > 0)*result._grad
        result._backward = _backward

        return result
    
    def cos(self):

        result = Tensor(math.cos(self.data), (self,), "cos")

        def _backward():
            self._grad += (math.sin(self.data))*result._grad
        result._backward = _backward

        return result

    def sin(self):

        result = Tensor(math.sin(self.data), (self,), "sin")

        def _backward():
            self._grad += (math.cos(self.data))*result._grad
        result._backward = _backward

        return result
    
    def tan(self):

        result = Tensor(math.tan(self.data), (self,), "tan")

        def _backward():
            self._grad += ((1/math.cos(self.data))**2)*result._grad
        result._backward = _backward

        return result

    def cot(self):

        result = Tensor(math.tan(self.data)**-1, (self,), "tan")

        def _backward():
            self._grad += (-1*(1/math.sin(self.data))**2)*result._grad
        result._backward = _backward

        return result

    def tanh(self):

        result = Tensor(math.tanh(self.data), (self,), "tan")

        def _backward():
            self._grad += ((1/math.cosh(self.data))**2)*result._grad
        result._backward = _backward

        return result    

    def log(self, base="e"):

        if base == "e":
            base = math.e
            log = lambda x : math.log(x)
        elif (isinstance(base, int) or isinstance(base, float)) and base > 0:
            log = lambda x : math.log(x, base)
        
        result = Tensor(log(self.data), (self,), "log")

        def _backward():
            self._grad = (1/(math.log(base)*self.data))*result._grad
        result._backward = _backward

        return result
    
    def exp(self):

        result = Tensor(math.exp(self.data), (self,), "exp")

        def _backward():
            self._grad = result.data*result._grad
        result._backward = _backward

        return result
    
    def abs(self):
        
        result = Tensor(abs(self.data), (self,), "abs")

        def _backward():
            
            if self.data == 0:
                _derivative = 0
            elif self.data < 0:
                _derivative = -1
            else:
                _derivative = 1
            
            self._grad += _derivative * result._grad

        result._backward = _backward

        return result
    
    def numpy(self, grad=True):
        """
        returns the tensor values when called.

        Note: doesn't actually return in numpy format. Returns a list of 
              the tensor.data and tensor.grad if grad = True. Else just returns 
              a list of tensor.data
        """
        if grad:
            return [self.data, self._grad]
        else:
            return [self.data]




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

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, var): # other + self
        return self + var

    def __sub__(self, var): # self - other
        return self + (-var)

    def __rsub__(self, var): # other - self
        return var + (-self)

    def __rmul__(self, var): # other * self
        return self * var

    def __truediv__(self, var): # self / other
        return self * var**-1

    def __rtruediv__(self, var): # other / self
        return var * self**-1




        

