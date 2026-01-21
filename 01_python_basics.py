# =============================================================================
# LESSON 1: Python Basics for PyTorch
# =============================================================================
# You know ML theory - now let's learn the code!
# Run this file line by line to see outputs.

# -----------------------------------------------------------------------------
# VARIABLES - storing values
# -----------------------------------------------------------------------------
learning_rate = 0.001      # A float (decimal number)
epochs = 100               # An integer (whole number)
model_name = "my_network"  # A string (text)
is_training = True         # A boolean (True/False)

print("Learning rate:", learning_rate)
print("Epochs:", epochs)

# -----------------------------------------------------------------------------
# LISTS - ordered collections (like arrays)
# -----------------------------------------------------------------------------
layer_sizes = [784, 256, 128, 10]  # Input -> Hidden -> Hidden -> Output
print("Layer sizes:", layer_sizes)
print("First layer size:", layer_sizes[0])   # Index starts at 0!
print("Last layer size:", layer_sizes[-1])   # Negative index = from end

# -----------------------------------------------------------------------------
# LOOPS - repeating actions
# -----------------------------------------------------------------------------
print("\nTraining epochs:")
for epoch in range(5):  # range(5) gives [0, 1, 2, 3, 4]
    print(f"  Epoch {epoch + 1}")  # f-string lets you embed variables

# Looping through a list
print("\nLayer dimensions:")
for size in layer_sizes:
    print(f"  {size} neurons")

# -----------------------------------------------------------------------------
# FUNCTIONS - reusable code blocks
# -----------------------------------------------------------------------------
def calculate_loss(predicted, actual):
    """This is a docstring - it describes what the function does."""
    error = predicted - actual
    squared_error = error ** 2  # ** means power/exponent
    return squared_error

# Using the function
loss = calculate_loss(0.8, 1.0)
print(f"\nLoss value: {loss}")

# -----------------------------------------------------------------------------
# CONDITIONALS - making decisions
# -----------------------------------------------------------------------------
accuracy = 0.95

if accuracy > 0.9:
    print("Excellent model performance!")
elif accuracy > 0.7:
    print("Good model performance")
else:
    print("Model needs improvement")

# -----------------------------------------------------------------------------
# EXERCISE: Try modifying the values above and re-running!
# -----------------------------------------------------------------------------
