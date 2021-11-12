import torch
print(torch.__version__)
import torch
import numpy as np

## tensors can eb created directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data.dtype) ## data type directly inferred
print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

######## The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

### shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")