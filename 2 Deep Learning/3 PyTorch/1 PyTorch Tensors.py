#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 PyTorch Tensors
# @date - 03/09/2018
#

import numpy as np
import torch

x = torch.rand(3, 2)
print(x)

y = torch.ones(x.size())
print(y)

z = x + y
print(z)

# In general PyTorch tensors behave similar to Numpy arrays. They are zero
# indexed and support slicing.
print(z[0])

print(z[:, 1:])

# Return a new tensor z + 1
print(z.add(1))

# z tensor is unchanged
print(z)

# Add 1 and update z tensor in-place
print(z.add_(1))

# z has been updated
print(z)

# Reshaping
print(z.size())

# Reshape the arrary permanently via the below
print(z.resize_(2, 3))

print(z)

# Numpy to Torch and back
a = np.random.rand(4, 3)
print(a)

b = torch.from_numpy(a)
print(b)

print(b.numpy())

# Multiply PyTorch Tensor by 2, in place
print(b.mul_(2))

# Numpy array matches new values from Tensor
print(a)
