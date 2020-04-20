from tinygrad.core import Tensor

x = Tensor(3)
y = Tensor(2)
z = x**4*y + x**3*y**2 + x**2*y**3 + x*y**4
z.backward() 
print(x)
print(y)