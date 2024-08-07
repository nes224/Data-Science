import torch
import numpy as np

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# a += 1
# print(a)
# print(b)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x + y
#     z = z.to("cpu")

x = torch.ones(5, requires_grad=True) # requires_grad tell pytorch that it will need to calculate the gradients for this tensor later in your optimization steps.
print(x)

