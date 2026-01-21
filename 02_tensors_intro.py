# =============================================================================
# LESSON 2: Introduction to PyTorch Tensors
# =============================================================================
# Tensors are PyTorch's fundamental data structure - like NumPy arrays but
# with GPU support and automatic differentiation built in.

import torch

# -----------------------------------------------------------------------------
# CREATING TENSORS
# -----------------------------------------------------------------------------

# From a Python list
data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data)
print("Tensor from list:")
print(tensor_from_list)

# Common initialization methods
zeros = torch.zeros(3, 4)      # 3x4 matrix of zeros
ones = torch.ones(2, 3)        # 2x3 matrix of ones
random = torch.rand(2, 2)      # 2x2 matrix, uniform random [0,1)
randn = torch.randn(2, 2)      # 2x2 matrix, normal distribution

print("\nZeros tensor (3x4):")
print(zeros)

print("\nRandom tensor:")
print(random)

# -----------------------------------------------------------------------------
# TENSOR ATTRIBUTES - understanding your data
# -----------------------------------------------------------------------------
x = torch.rand(3, 4)

print(f"\nTensor shape: {x.shape}")      # Dimensions: torch.Size([3, 4])
print(f"Tensor dtype: {x.dtype}")        # Data type: torch.float32
print(f"Tensor device: {x.device}")      # Where it lives: cpu or cuda

# -----------------------------------------------------------------------------
# RESHAPING - changing tensor dimensions
# -----------------------------------------------------------------------------
original = torch.arange(12)  # Creates [0, 1, 2, ..., 11]
print(f"\nOriginal 1D tensor: {original}")

reshaped = original.reshape(3, 4)  # Reshape to 3x4
print("Reshaped to 3x4:")
print(reshaped)

flattened = reshaped.flatten()  # Back to 1D
print(f"Flattened: {flattened}")

# -----------------------------------------------------------------------------
# BASIC OPERATIONS - element-wise math
# -----------------------------------------------------------------------------
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")           # Element-wise addition
print(f"a * b = {a * b}")           # Element-wise multiplication
print(f"a @ b = {a @ b}")           # Dot product (scalar)

# Matrix multiplication
matrix1 = torch.rand(2, 3)
matrix2 = torch.rand(3, 4)
result = matrix1 @ matrix2  # Result is 2x4
print(f"\nMatrix multiplication: {matrix1.shape} @ {matrix2.shape} = {result.shape}")

# -----------------------------------------------------------------------------
# INDEXING AND SLICING - accessing parts of tensors
# -----------------------------------------------------------------------------
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(f"\nFull tensor:\n{x}")
print(f"First row: {x[0]}")           # [1, 2, 3]
print(f"First column: {x[:, 0]}")     # [1, 4, 7]
print(f"Element at [1,2]: {x[1, 2]}") # 6
print(f"Last row: {x[-1]}")           # [7, 8, 9]

# -----------------------------------------------------------------------------
# GPU SUPPORT - moving tensors to GPU (if available)
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    gpu_tensor = torch.rand(3, 3, device=device)
    print(f"\nGPU tensor device: {gpu_tensor.device}")
else:
    print("\nNo GPU available, using CPU")

# -----------------------------------------------------------------------------
# EXERCISE: Create a 5x5 identity matrix using torch.eye()
# Then multiply it with a random 5x5 matrix. What do you expect?
# -----------------------------------------------------------------------------
