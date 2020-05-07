from tinygrad import Tensor
from tinygrad.nn import Layer, SimpleMLP

layers = [2,2]
model = SimpleMLP(1,1,layers)
#model.summary()

x = [Tensor(5)]
out = model(x)
print(out)
out.backward()

print(out)