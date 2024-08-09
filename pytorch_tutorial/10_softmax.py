import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# x = np.array([2.0, 1.0, 0.1])
# outputs = softmax(x)
# print('softmax numpy:', outputs)

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

'''
Cross-Entropy 
cross entropy loss so this measures the performance of our classification model whose
output is a probability between 0 & 1 
Y = [1,0,0] One-Hot Encoded Class Labels
^
Y = [0.7, 0.2, 0.1] Probabilities (Softmaxo)
'''