import torch
import torch.nn as nn

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # If you don't know what to use, just use a ReLU for hidden layers.
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        y_pred = torch.signmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28,hidden_size=5)
criterion = nn.BCELoss()

# TanH Function -> Hidden layers.
# Leaky ReLU Function -> Improved version of ReLU Tries to solve the vanishing gradient problem.
# Softmax -> Good in last layer in multi class classification problems.