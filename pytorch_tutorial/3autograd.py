import torch

x = torch.randn(3, requires_grad=True) # if we don't specify this argument -> requires_grad -> they don't have this grad function attribute.
print(x)
"""
Forward pass
------->
x
  \
    --- (+) ---y -
  /              |   grad-fn
2            <-----
           Add backward -> calculate the gradient of Y with respect to X
           dy/dx
"""
y = x+2
print(y)
z = y*y*2
# z = z.mean() # grad can be implicitly created only for scalar outputs. z.mean() scalar values.
print(z)

# when we want to calculate the gradients the only thing that 
# we must do is to call backward()
v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32) # scalar values.
z.backward(v) # dz/dx -> vector Jacobian product to get the gradients.
print(x.grad)