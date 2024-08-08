"""
Backpropagation
Chain Rule: az/az = az/ay * ay/ax

Computational Graph
x
  \
   ------ f --- z -- > 0 -> 0 -> Loss
  /
y
( loss function )
f = x * y 
local gradients
dz/dx = df/dx = y
dz/dy = df/dy = x
dLoss/dx
dLoss/dx = dLoss/dz * dz/dx

1) Forward pass: Compute Loss
2) Compute local gradients (learning rate)
3) Backward pass: Compute dLoss / dWeights using the Chain Rule
 ^                  ^
 y = w * x, loss = (y - y)^2 = (wx-y)^2
x
 \       ^
  -- (*) y -- (-) -learning rate- ()
 /          /
w          y

y hat = y predicted

Minimize loss -> aLoss/aw

"""

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass and cumpute the loss
y_hat = w * x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)

### update weights
### next forward and backwards



