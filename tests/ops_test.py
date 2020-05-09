from tinygrad import Tensor 

# we manually test the values of the mathematical operations
# we do this by comparing the computing the forward and backward values
# and comparing them to the values calculated manually

# add
a = Tensor(5)
b = Tensor(5)
c = a + b
d = a - b
c.backward()
for idx, t, exp_grad in [("a",a,1), ("b",b,1), ("c",c,1)]:
    print("For {}, value: {}, expected_grad: {}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

# subtract
a = Tensor(5)
b = Tensor(5)
c = a - b
c.backward()
for idx, t, exp_grad in [("a", a, 1), ("b",b, -1), ("c", c, 1)]:
    print("For {}, value: {}, expected_grad:{}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

# multiply
a = Tensor(5)
b = Tensor(5)
c = a * b
c.backward()
for idx, t, exp_grad in [("a",a,5), ("b",b,5), ("c",c,1)]:
    print("For {}, value: {}, expected_grad:{}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

# division (also pow)
a = Tensor(5)
b = Tensor(10)
c = a / (b**3)
c.backward()
for idx, t, exp_grad in [("a", a, 0.001), ("b", b, -0.0015), ("c", c, 1)]:
    print("For {}, value: {}, expected_grad:{}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

# trignometric functions
a = Tensor(45)
b = Tensor(45)
c = Tensor(45)
d = Tensor(45)

_a = a.sin()
_a.backward()
_b = b.cos()
_b.backward()
_c = c.tan()
_c.backward()
_d = d.tanh()
_d.backward()
for idx, t, exp_grad in [("a", a, 0.52532199), ("b", b, 0.85090352), ("c", c, 3.6236716679014025), ("d", d, 3.277605049596206e-39)]:
    print("For {}, value: {}, expected_grad:{}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

# log
a = Tensor(2)
b = Tensor(2)
c = Tensor(2)
_a = a.log()
_b = b.exp()
_c = c.log(15)
_a.backward()
_b.backward()
_c.backward()
for idx, t, exp_grad in [("a", a, .5), ("b", b, 7.3890560989306495), ("c", c, 0.18463468653442752)]:
    print("For {}, value: {}, expected_grad:{}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

#abs
a = Tensor(50)
b = Tensor(-50.123123123)
c = Tensor(0)
_a = a.abs()
_b = b.abs()
_c = c.abs()
_a.backward()
_b.backward()
_c.backward()
for idx, t, exp_grad in [("a", a, 1), ("b", b, -1), ("c", c, 0)]:
    print("For {}, value: {}, expected_grad:{}, grad: {}".format(idx, t.data, exp_grad, t._grad))
print("-"*50)

