#!/usr/bin/env python3
"""
================================================================================
LESSON 02: Introduction to PyTorch Tensors
================================================================================
Interactive lesson on tensors - PyTorch's fundamental data structure.
Tensors are like NumPy arrays but with GPU support and autograd!

Run: python 02_tensors_intro.py
================================================================================
"""

import sys
sys.path.insert(0, '.')

import torch
from utils import (
    header, subheader, success, error, info, warning,
    ExerciseTest, LessonRunner, ask_question,
    test_equal, test_shape, test_true
)

# Create the lesson runner
lesson = LessonRunner(
    "02_tensors",
    "Master PyTorch tensors - the foundation of all deep learning in PyTorch."
)

# =============================================================================
# SECTION 1: Creating Tensors
# =============================================================================

@lesson.section("Creating Tensors")
def section_creating():
    """Learn different ways to create tensors."""

    print("Tensors are the core data structure in PyTorch.\n")
    print("Think of them as multi-dimensional arrays with superpowers:\n")
    print("  - GPU acceleration")
    print("  - Automatic differentiation")
    print("  - Broadcasting operations\n")

    print("-" * 50)
    print("FROM PYTHON DATA:")
    print("-" * 50)

    # From a list
    data = [[1, 2], [3, 4]]
    t = torch.tensor(data)
    print(f"\ntorch.tensor([[1, 2], [3, 4]]):")
    print(t)

    # From a single value (scalar)
    scalar = torch.tensor(3.14)
    print(f"\ntorch.tensor(3.14) = {scalar}  (0-dimensional tensor)")

    print("\n" + "-" * 50)
    print("COMMON INITIALIZATIONS:")
    print("-" * 50)

    zeros = torch.zeros(2, 3)
    print(f"\ntorch.zeros(2, 3):  # 2 rows, 3 cols of zeros")
    print(zeros)

    ones = torch.ones(2, 3)
    print(f"\ntorch.ones(2, 3):  # 2 rows, 3 cols of ones")
    print(ones)

    rand = torch.rand(2, 3)
    print(f"\ntorch.rand(2, 3):  # Uniform random [0, 1)")
    print(rand)

    randn = torch.randn(2, 3)
    print(f"\ntorch.randn(2, 3):  # Normal distribution (mean=0, std=1)")
    print(randn)

    arange = torch.arange(0, 10, 2)
    print(f"\ntorch.arange(0, 10, 2) = {arange}  # Like range(), but tensor")

    eye = torch.eye(3)
    print(f"\ntorch.eye(3):  # Identity matrix")
    print(eye)

    print("\n" + "-" * 50)
    print("KEY INSIGHT: Tensor dimensions are (batch, channels, height, width)")
    print("or (batch, sequence, features) for NLP. Batch comes first!")
    print("-" * 50)


@lesson.exercise("Create a Tensor", points=1)
def exercise_create():
    """Test tensor creation."""

    # Student task: Create a 3x4 tensor of zeros
    student_tensor = torch.zeros(3, 4)

    test = ExerciseTest("Create a 3x4 zeros tensor")
    test.check_shape(student_tensor, (3, 4), "Shape should be (3, 4)")
    test.check_true(
        torch.all(student_tensor == 0).item(),
        "All values should be 0"
    )
    return test.run()


# =============================================================================
# SECTION 2: Tensor Attributes
# =============================================================================

@lesson.section("Tensor Attributes")
def section_attributes():
    """Learn about tensor properties: shape, dtype, device."""

    x = torch.rand(3, 4)
    print("x = torch.rand(3, 4)\n")

    print(f"x.shape  = {x.shape}      # Dimensions (torch.Size)")
    print(f"x.dtype  = {x.dtype}  # Data type")
    print(f"x.device = {x.device}        # Where tensor lives")
    print(f"x.ndim   = {x.ndim}             # Number of dimensions")
    print(f"x.numel()= {x.numel()}            # Total number of elements")

    print("\n" + "-" * 50)
    print("COMMON DATA TYPES:")
    print("-" * 50)
    print("\n  torch.float32 (default) - Standard for neural nets")
    print("  torch.float64          - Double precision")
    print("  torch.float16          - Half precision (faster on GPU)")
    print("  torch.int32            - Integer")
    print("  torch.int64 (long)     - Long integer (for indices)")
    print("  torch.bool             - Boolean")

    print("\n" + "-" * 50)
    print("SPECIFYING DTYPE:")
    print("-" * 50)

    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    print(f"\ntorch.tensor([1, 2, 3], dtype=torch.int64) -> dtype: {int_tensor.dtype}")

    float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(f"torch.tensor([1, 2, 3], dtype=torch.float32) -> dtype: {float_tensor.dtype}")

    print("\n" + "-" * 50)
    print("DEVICE (CPU vs GPU):")
    print("-" * 50)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_tensor = torch.rand(2, 2, device='cuda')
        print(f"GPU tensor device: {gpu_tensor.device}")
    else:
        print("No GPU available, using CPU")
        print("To move tensor to GPU: tensor.to('cuda')")


@lesson.exercise("Tensor Attributes Quiz", points=1)
def exercise_attributes():
    """Test understanding of tensor attributes."""

    x = torch.randn(2, 3, 4)

    answer = ask_question(
        f"A tensor has shape {x.shape}. How many total elements does it have?",
        ["9", "24", "12", "6"]
    )

    test = ExerciseTest("Tensor Size", hint="Multiply all dimensions: 2 * 3 * 4")
    test.check_true(answer == 1, f"Correct! 2 × 3 × 4 = 24 (x.numel() = {x.numel()})")
    return test.run()


# =============================================================================
# SECTION 3: Indexing and Slicing
# =============================================================================

@lesson.section("Indexing and Slicing")
def section_indexing():
    """Learn to access parts of tensors - crucial for data manipulation."""

    x = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    print("x = ")
    print(x)
    print()

    print("-" * 50)
    print("BASIC INDEXING:")
    print("-" * 50)
    print(f"\nx[0]     = {x[0]}        # First row")
    print(f"x[0, 0]  = {x[0, 0]}                   # Element at row 0, col 0")
    print(f"x[-1]    = {x[-1]}       # Last row")
    print(f"x[-1, -1]= {x[-1, -1]}                  # Last element")

    print("\n" + "-" * 50)
    print("SLICING [start:end:step]:")
    print("-" * 50)
    print(f"\nx[:, 0]    = {x[:, 0]}       # All rows, first column")
    print(f"x[:, -1]   = {x[:, -1]}      # All rows, last column")
    print(f"x[0, :]    = {x[0, :]}    # First row, all columns")
    print(f"x[0:2, :]  = \n{x[0:2, :]}     # First 2 rows")
    print(f"x[:, 1:3]  = \n{x[:, 1:3]}     # Columns 1 and 2")

    print("\n" + "-" * 50)
    print("ADVANCED INDEXING:")
    print("-" * 50)
    indices = torch.tensor([0, 2])
    print(f"\nindices = {indices}")
    print(f"x[indices] = \n{x[indices]}  # Select rows 0 and 2")

    mask = x > 5
    print(f"\nmask = (x > 5):")
    print(mask)
    print(f"x[mask] = {x[mask]}  # Elements where mask is True")

    print("\n" + "-" * 50)
    print("KEY INSIGHT: Slicing creates a VIEW, not a copy!")
    print("Modifying the slice modifies the original tensor.")
    print("-" * 50)


@lesson.exercise("Tensor Slicing", points=1)
def exercise_indexing():
    """Test understanding of tensor indexing."""

    x = torch.arange(12).reshape(3, 4)
    print(f"x = \n{x}\n")

    answer = ask_question(
        "What does x[:, 2] return?",
        ["tensor([2, 6, 10])", "tensor([8, 9, 10, 11])", "tensor([0, 4, 8])", "tensor([2, 3])"]
    )

    test = ExerciseTest("Slicing", hint="[:, 2] means all rows, column index 2")
    expected = x[:, 2]
    test.check_true(answer == 0, f"Correct! x[:, 2] = {expected} (all rows, column 2)")
    return test.run()


# =============================================================================
# SECTION 4: Reshaping Tensors
# =============================================================================

@lesson.section("Reshaping Tensors")
def section_reshaping():
    """Learn to change tensor shapes - essential for neural network layers."""

    print("Reshaping changes the arrangement of elements without changing them.\n")

    x = torch.arange(12)
    print(f"x = torch.arange(12) = {x}")
    print(f"x.shape = {x.shape}\n")

    print("-" * 50)
    print("RESHAPE:")
    print("-" * 50)

    reshaped = x.reshape(3, 4)
    print(f"\nx.reshape(3, 4) =")
    print(reshaped)

    reshaped2 = x.reshape(2, 2, 3)
    print(f"\nx.reshape(2, 2, 3) =")
    print(reshaped2)

    auto_shape = x.reshape(3, -1)  # -1 means "infer this dimension"
    print(f"\nx.reshape(3, -1) =  # -1 is inferred as 4")
    print(auto_shape)

    print("\n" + "-" * 50)
    print("VIEW vs RESHAPE:")
    print("-" * 50)
    print("\n.view()    - Returns a view (shares memory, must be contiguous)")
    print(".reshape() - More flexible, may copy if needed")
    print(".contiguous().view() - Safe way to use view")

    print("\n" + "-" * 50)
    print("OTHER RESHAPING OPERATIONS:")
    print("-" * 50)

    t = torch.rand(2, 3)
    print(f"\nt = torch.rand(2, 3) with shape {t.shape}")

    print(f"t.flatten()        -> shape {t.flatten().shape}  # To 1D")
    print(f"t.unsqueeze(0)     -> shape {t.unsqueeze(0).shape}  # Add dim at position 0")
    print(f"t.unsqueeze(-1)    -> shape {t.unsqueeze(-1).shape}  # Add dim at end")

    t2 = t.unsqueeze(0)
    print(f"t2.squeeze()       -> shape {t2.squeeze().shape}  # Remove size-1 dims")

    print(f"\nt.T                -> shape {t.T.shape}  # Transpose")
    print(f"t.permute(1, 0)    -> shape {t.permute(1, 0).shape}  # Reorder dimensions")

    print("\n" + "-" * 50)
    print("KEY INSIGHT: Total elements must stay the same!")
    print(f"12 elements can be: (12,), (3,4), (4,3), (2,6), (2,2,3), etc.")
    print("-" * 50)


@lesson.exercise("Reshape Quiz", points=1)
def exercise_reshape():
    """Test understanding of reshaping."""

    x = torch.rand(4, 6)

    answer = ask_question(
        f"Can a tensor with shape (4, 6) be reshaped to (8, 3)?",
        ["Yes", "No - different number of elements", "Only with .view()", "Only on GPU"]
    )

    test = ExerciseTest("Reshape", hint="Count total elements: 4×6 vs 8×3")
    test.check_true(answer == 0, "Correct! 4×6 = 24 = 8×3, so reshape is valid")

    # Demonstrate
    reshaped = x.reshape(8, 3)
    print(f"\nDemonstration: {x.shape} -> {reshaped.shape}")
    return test.run()


# =============================================================================
# SECTION 5: Basic Operations
# =============================================================================

@lesson.section("Basic Operations")
def section_operations():
    """Learn tensor math operations."""

    print("Tensors support all standard math operations.\n")

    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5., 6.])

    print(f"a = {a}")
    print(f"b = {b}\n")

    print("-" * 50)
    print("ELEMENT-WISE OPERATIONS:")
    print("-" * 50)

    print(f"\na + b  = {a + b}        # Addition")
    print(f"a - b  = {a - b}       # Subtraction")
    print(f"a * b  = {a * b}        # Element-wise multiply")
    print(f"a / b  = {a / b}  # Division")
    print(f"a ** 2 = {a ** 2}        # Power")

    print("\n" + "-" * 50)
    print("MATRIX OPERATIONS:")
    print("-" * 50)

    A = torch.rand(2, 3)
    B = torch.rand(3, 4)

    print(f"\nA.shape = {A.shape}")
    print(f"B.shape = {B.shape}")
    print(f"A @ B shape = {(A @ B).shape}  # Matrix multiplication")

    print("\n# Dot product of vectors:")
    print(f"a @ b = {a @ b}  # Same as (a * b).sum()")

    print("\n" + "-" * 50)
    print("AGGREGATION OPERATIONS:")
    print("-" * 50)

    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    print(f"\nx = \n{x}")

    print(f"\nx.sum()     = {x.sum()}")
    print(f"x.mean()    = {x.mean()}")
    print(f"x.max()     = {x.max()}")
    print(f"x.min()     = {x.min()}")
    print(f"x.std()     = {x.std():.4f}")

    print(f"\nx.sum(dim=0)  = {x.sum(dim=0)}  # Sum along rows (collapse dim 0)")
    print(f"x.sum(dim=1)  = {x.sum(dim=1)}     # Sum along cols (collapse dim 1)")

    print(f"\nx.argmax()    = {x.argmax()}         # Index of max (flattened)")
    print(f"x.argmax(dim=1)= {x.argmax(dim=1)}      # Index of max per row")


@lesson.exercise("Matrix Multiplication", points=1)
def exercise_operations():
    """Test understanding of matrix operations."""

    A = torch.rand(5, 3)
    B = torch.rand(3, 7)

    answer = ask_question(
        f"What is the shape of A @ B where A is (5,3) and B is (3,7)?",
        ["(3, 3)", "(5, 7)", "(5, 3, 7)", "Error - incompatible shapes"]
    )

    test = ExerciseTest("Matrix Multiplication", hint="(m,k) @ (k,n) = (m,n)")
    result = A @ B
    test.check_true(answer == 1, f"Correct! (5,3) @ (3,7) = {tuple(result.shape)}")
    return test.run()


# =============================================================================
# SECTION 6: Broadcasting
# =============================================================================

@lesson.section("Broadcasting")
def section_broadcasting():
    """Learn how PyTorch handles operations on different-shaped tensors."""

    print("Broadcasting: automatic expansion for element-wise operations.\n")

    x = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])
    y = torch.tensor([10., 20., 30.])

    print(f"x shape: {x.shape}")
    print(x)
    print(f"\ny shape: {y.shape}")
    print(y)

    result = x + y
    print(f"\nx + y shape: {result.shape}")
    print(result)
    print("\n(y was broadcast across all rows of x)")

    print("\n" + "-" * 50)
    print("BROADCASTING RULES:")
    print("-" * 50)
    print("\n1. Align shapes from the right")
    print("2. Dimensions must be equal OR one of them is 1")
    print("3. Size-1 dimensions are 'stretched' to match")

    print("\nExamples of compatible shapes:")
    print("  (3, 4) + (4,)   -> (3, 4)  # y broadcast to each row")
    print("  (3, 4) + (3, 1) -> (3, 4)  # y broadcast to each col")
    print("  (3, 1) + (1, 4) -> (3, 4)  # Both broadcast!")

    # Demonstrate
    a = torch.ones(3, 1)
    b = torch.ones(1, 4)
    print(f"\n(3, 1) + (1, 4) =")
    print(f"Shapes: {a.shape} + {b.shape} = {(a + b).shape}")

    print("\n" + "-" * 50)
    print("KEY INSIGHT: Broadcasting is memory efficient!")
    print("No actual copying happens - operations are computed on-the-fly.")
    print("-" * 50)


@lesson.exercise("Broadcasting Quiz", points=1)
def exercise_broadcasting():
    """Test understanding of broadcasting."""

    answer = ask_question(
        "Can you add tensors with shapes (4, 1, 3) and (5, 3)?",
        ["Yes, result is (4, 5, 3)", "Yes, result is (4, 1, 3)", "No - incompatible shapes", "Yes, result is (5, 3)"]
    )

    test = ExerciseTest("Broadcasting", hint="Align from right: (4,1,3) and (5,3) -> (4,5,3)")

    # Demonstrate
    a = torch.rand(4, 1, 3)
    b = torch.rand(5, 3)
    result = a + b

    test.check_true(answer == 0, f"Correct! {a.shape} + {b.shape} = {result.shape}")
    return test.run()


# =============================================================================
# SECTION 7: GPU Operations
# =============================================================================

@lesson.section("GPU Operations")
def section_gpu():
    """Learn to move tensors between CPU and GPU."""

    print("PyTorch makes GPU acceleration easy!\n")

    print(f"CUDA (GPU) available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    print("\n" + "-" * 50)
    print("MOVING TENSORS:")
    print("-" * 50)

    x = torch.rand(3, 3)
    print(f"\nx = torch.rand(3, 3)")
    print(f"x.device = {x.device}")

    print("\n# Move to GPU (if available):")
    print("x_gpu = x.to('cuda')     # or x.cuda()")
    print("x_cpu = x_gpu.to('cpu')  # or x_gpu.cpu()")

    print("\n# Create directly on GPU:")
    print("y = torch.rand(3, 3, device='cuda')")

    print("\n" + "-" * 50)
    print("DEVICE-AGNOSTIC CODE:")
    print("-" * 50)

    print("""
# Best practice - works on any hardware:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

x = torch.rand(3, 3).to(device)
model = MyModel().to(device)
""")

    print("-" * 50)
    print("KEY INSIGHT: All tensors in an operation must be on the same device!")
    print("cpu_tensor + gpu_tensor = ERROR")
    print("-" * 50)

    # Demonstrate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nCurrent best device: {device}")


@lesson.exercise("Device Quiz", points=1)
def exercise_gpu():
    """Test understanding of devices."""

    answer = ask_question(
        "What happens if you try: torch.rand(3, device='cuda') + torch.rand(3, device='cpu')?",
        ["Result is on CPU", "Result is on GPU", "Runtime Error", "Result is automatically moved"]
    )

    test = ExerciseTest("Device Operations", hint="PyTorch does NOT auto-move between devices")
    test.check_true(answer == 2, "Correct! Tensors must be on the same device for operations")
    return test.run()


# =============================================================================
# FINAL CODING EXERCISE
# =============================================================================

@lesson.exercise("Coding Challenge: Normalize Data", points=3)
def exercise_final():
    """Final coding exercise combining multiple concepts."""

    print("Let's normalize a batch of data (common preprocessing step)!\n")
    print("Task: Given a batch of data, normalize each feature to mean=0, std=1")
    print()
    print("Formula: normalized = (x - mean) / std")
    print("Compute mean and std along the batch dimension (dim=0)")

    # Create sample data: batch of 5 samples, 3 features each
    torch.manual_seed(42)
    data = torch.rand(5, 3) * 10  # Values 0-10

    print(f"\ndata = \n{data}")
    print(f"data.shape = {data.shape}")

    # Solution
    mean = data.mean(dim=0)  # Mean of each feature
    std = data.std(dim=0)    # Std of each feature
    normalized = (data - mean) / std

    print(f"\nmean (per feature) = {mean}")
    print(f"std (per feature) = {std}")
    print(f"\nnormalized = \n{normalized}")

    # Verify
    test = ExerciseTest("Data Normalization")

    # Check that mean is ~0 for each feature
    new_mean = normalized.mean(dim=0)
    test.check_true(
        torch.allclose(new_mean, torch.zeros(3), atol=1e-5),
        f"Normalized mean ≈ 0: {new_mean}"
    )

    # Check that std is ~1 for each feature (with Bessel's correction)
    new_std = normalized.std(dim=0)
    test.check_true(
        torch.allclose(new_std, torch.ones(3), atol=0.2),
        f"Normalized std ≈ 1: {new_std}"
    )

    # Check shape preserved
    test.check_shape(normalized, (5, 3), "Shape preserved after normalization")

    return test.run()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check for --test flag for non-interactive mode
    if "--test" in sys.argv:
        lesson.run()
    else:
        lesson.run_interactive()
