#!/usr/bin/env python3
"""
================================================================================
LESSON 01: Python Basics for PyTorch
================================================================================
Interactive introduction to Python fundamentals needed for PyTorch.
Skip this if you're already comfortable with Python!

Run: python 01_python_basics.py
================================================================================
"""

import sys
sys.path.insert(0, '.')

from utils import (
    header, subheader, success, error, info, warning,
    ExerciseTest, LessonRunner, ask_question,
    test_equal, test_true
)

# Create the lesson runner
lesson = LessonRunner(
    "01_python_basics",
    "Learn the Python fundamentals needed for PyTorch programming."
)

# =============================================================================
# SECTION 1: Variables and Data Types
# =============================================================================

@lesson.section("Variables and Data Types")
def section_variables():
    """Learn about Python variables and basic data types."""

    print("In Python, variables are created by assignment (no type declaration needed):\n")

    # Demonstrate variables
    learning_rate = 0.001      # float
    epochs = 100               # int
    model_name = "my_network"  # str
    is_training = True         # bool

    print(f"learning_rate = 0.001      # float: {type(learning_rate).__name__}")
    print(f"epochs = 100               # int: {type(epochs).__name__}")
    print(f'model_name = "my_network"  # str: {type(model_name).__name__}')
    print(f"is_training = True         # bool: {type(is_training).__name__}")

    print("\n" + "-"*50)
    print("KEY INSIGHT: Python figures out types automatically!")
    print("This is called 'dynamic typing'.")
    print("-"*50)

    # Type conversion
    print("\nType conversion:")
    x = 5
    y = float(x)  # Convert int to float
    z = str(x)    # Convert int to string
    print(f"  int(5) = {x}")
    print(f"  float(5) = {y}")
    print(f"  str(5) = '{z}'")


@lesson.exercise("Variables Quiz", points=1)
def exercise_variables():
    """Test understanding of variables."""

    answer = ask_question(
        "What is the type of the variable: x = 3.14?",
        ["int", "float", "str", "bool"]
    )

    test = ExerciseTest("Variables Quiz", hint="3.14 has a decimal point!")
    test.check_true(answer == 1, "Correct! 3.14 is a float (decimal number)")
    return test.run()


# =============================================================================
# SECTION 2: Lists and Indexing
# =============================================================================

@lesson.section("Lists and Indexing")
def section_lists():
    """Learn about Python lists - the foundation for tensor understanding."""

    print("Lists are ordered collections (similar to arrays in other languages):\n")

    layer_sizes = [784, 256, 128, 10]
    print(f"layer_sizes = {layer_sizes}")

    print("\nIndexing (0-based!):")
    print(f"  layer_sizes[0]  = {layer_sizes[0]}   # First element")
    print(f"  layer_sizes[1]  = {layer_sizes[1]}   # Second element")
    print(f"  layer_sizes[-1] = {layer_sizes[-1]}    # Last element")
    print(f"  layer_sizes[-2] = {layer_sizes[-2]}   # Second to last")

    print("\nSlicing [start:end] (end is exclusive):")
    print(f"  layer_sizes[0:2]  = {layer_sizes[0:2]}    # First two")
    print(f"  layer_sizes[1:]   = {layer_sizes[1:]}  # All except first")
    print(f"  layer_sizes[:2]   = {layer_sizes[:2]}     # First two")
    print(f"  layer_sizes[::2]  = {layer_sizes[::2]}    # Every other")

    print("\n" + "-"*50)
    print("KEY INSIGHT: This indexing works the same way in PyTorch tensors!")
    print("-"*50)

    print("\nList operations:")
    numbers = [1, 2, 3]
    numbers.append(4)
    print(f"  [1, 2, 3].append(4) -> {numbers}")

    combined = [1, 2] + [3, 4]
    print(f"  [1, 2] + [3, 4] -> {combined}")

    print(f"  len([1, 2, 3, 4]) = {len([1, 2, 3, 4])}")


@lesson.exercise("List Indexing", points=1)
def exercise_lists():
    """Test understanding of list indexing."""

    data = [10, 20, 30, 40, 50]

    answer = ask_question(
        f"Given data = {data}, what is data[-2]?",
        ["30", "40", "50", "Error"]
    )

    test = ExerciseTest("List Indexing", hint="-2 means second from the end")
    test.check_true(answer == 1, "Correct! data[-2] = 40 (second from end)")
    return test.run()


# =============================================================================
# SECTION 3: Loops
# =============================================================================

@lesson.section("Loops")
def section_loops():
    """Learn about for loops and iteration."""

    print("For loops iterate over sequences:\n")

    print("# Loop with range()")
    print("for i in range(5):")
    print("    print(i)")
    print("\nOutput:")
    for i in range(5):
        print(f"  {i}")

    print("\n# Loop over a list")
    print("layers = [128, 64, 32]")
    print("for size in layers:")
    print("    print(f'Layer with {size} neurons')")
    print("\nOutput:")
    layers = [128, 64, 32]
    for size in layers:
        print(f"  Layer with {size} neurons")

    print("\n# Loop with index using enumerate()")
    print("for i, size in enumerate(layers):")
    print("    print(f'Layer {i}: {size} neurons')")
    print("\nOutput:")
    for i, size in enumerate(layers):
        print(f"  Layer {i}: {size} neurons")

    print("\n" + "-"*50)
    print("KEY INSIGHT: range(5) gives [0, 1, 2, 3, 4] - NOT including 5!")
    print("This matches 0-based indexing perfectly.")
    print("-"*50)


@lesson.exercise("Loop Output", points=1)
def exercise_loops():
    """Test understanding of loops."""

    answer = ask_question(
        "What does range(3) produce?",
        ["[1, 2, 3]", "[0, 1, 2]", "[0, 1, 2, 3]", "[1, 2]"]
    )

    test = ExerciseTest("Loop Output", hint="range() starts at 0 and excludes the end")
    test.check_true(answer == 1, "Correct! range(3) = [0, 1, 2]")
    return test.run()


# =============================================================================
# SECTION 4: Functions
# =============================================================================

@lesson.section("Functions")
def section_functions():
    """Learn about defining and using functions."""

    print("Functions are defined with 'def' and can return values:\n")

    print("def calculate_loss(predicted, actual):")
    print('    """Calculate squared error loss."""')
    print("    error = predicted - actual")
    print("    return error ** 2")

    def calculate_loss(predicted, actual):
        """Calculate squared error loss."""
        error = predicted - actual
        return error ** 2

    print("\n# Using the function:")
    result = calculate_loss(0.8, 1.0)
    print(f"calculate_loss(0.8, 1.0) = {result}")

    print("\n# Functions with default arguments:")
    print("def train(model, epochs=10, lr=0.001):")
    print("    ...")

    def train(model, epochs=10, lr=0.001):
        return f"Training {model} for {epochs} epochs with lr={lr}"

    print(f"\ntrain('ResNet')         -> {train('ResNet')}")
    print(f"train('ResNet', 20)     -> {train('ResNet', 20)}")
    print(f"train('ResNet', lr=0.1) -> {train('ResNet', lr=0.1)}")

    print("\n" + "-"*50)
    print("KEY INSIGHT: Functions in PyTorch often have many default arguments.")
    print("You'll see: model.train(), optimizer.step(), etc.")
    print("-"*50)


@lesson.exercise("Function Output", points=1)
def exercise_functions():
    """Test understanding of functions."""

    def mystery(x, y=2):
        return x ** y

    answer = ask_question(
        "What does mystery(3) return?",
        ["3", "6", "9", "Error - missing argument"]
    )

    test = ExerciseTest("Function Output", hint="y has a default value of 2")
    test.check_true(answer == 2, "Correct! mystery(3) = 3**2 = 9")
    return test.run()


# =============================================================================
# SECTION 5: Conditionals
# =============================================================================

@lesson.section("Conditionals")
def section_conditionals():
    """Learn about if/elif/else statements."""

    print("Conditionals control program flow based on conditions:\n")

    accuracy = 0.85

    print(f"accuracy = {accuracy}")
    print()
    print("if accuracy > 0.9:")
    print('    print("Excellent!")')
    print("elif accuracy > 0.7:")
    print('    print("Good!")')
    print("else:")
    print('    print("Needs work")')

    print("\nOutput:")
    if accuracy > 0.9:
        print("  Excellent!")
    elif accuracy > 0.7:
        print("  Good!")
    else:
        print("  Needs work")

    print("\n# Comparison operators:")
    print("  ==  Equal to")
    print("  !=  Not equal to")
    print("  >   Greater than")
    print("  <   Less than")
    print("  >=  Greater than or equal")
    print("  <=  Less than or equal")

    print("\n# Logical operators:")
    print("  and  Both conditions must be true")
    print("  or   At least one condition must be true")
    print("  not  Inverts the condition")

    print("\nExample:")
    x, y = 5, 10
    print(f"  x = {x}, y = {y}")
    print(f"  x > 0 and y > 0: {x > 0 and y > 0}")
    print(f"  x > 10 or y > 5: {x > 10 or y > 5}")
    print(f"  not (x == y):    {not (x == y)}")


@lesson.exercise("Conditional Logic", points=1)
def exercise_conditionals():
    """Test understanding of conditionals."""

    loss = 0.05
    epochs = 100

    answer = ask_question(
        f"Given loss={loss} and epochs={epochs}, what is (loss < 0.1 and epochs >= 50)?",
        ["True", "False", "Error", "None"]
    )

    test = ExerciseTest("Conditional Logic", hint="Both conditions must be true for 'and'")
    test.check_true(answer == 0, "Correct! Both 0.05 < 0.1 and 100 >= 50 are True")
    return test.run()


# =============================================================================
# SECTION 6: Dictionaries
# =============================================================================

@lesson.section("Dictionaries")
def section_dictionaries():
    """Learn about dictionaries - key-value stores used everywhere in PyTorch."""

    print("Dictionaries store key-value pairs (like model configs!):\n")

    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "model": "resnet18"
    }

    print("config = {")
    for key, value in config.items():
        print(f'    "{key}": {repr(value)},')
    print("}")

    print("\nAccessing values:")
    print(f"  config['learning_rate'] = {config['learning_rate']}")
    print(f"  config.get('dropout', 0.5) = {config.get('dropout', 0.5)}  # Default if missing")

    print("\nModifying:")
    config["dropout"] = 0.2
    print(f"  config['dropout'] = 0.2  # Add new key")
    print(f"  config now has: {list(config.keys())}")

    print("\nIterating:")
    print("  for key, value in config.items():")
    print("      print(f'{key}: {value}')")

    print("\n" + "-"*50)
    print("KEY INSIGHT: PyTorch uses dicts everywhere!")
    print("  - model.state_dict()  # Model weights")
    print("  - {'loss': 0.5, 'acc': 0.9}  # Metrics")
    print("-"*50)


@lesson.exercise("Dictionary Access", points=1)
def exercise_dictionaries():
    """Test understanding of dictionaries."""

    params = {"lr": 0.01, "momentum": 0.9}

    answer = ask_question(
        "What does params.get('weight_decay', 0.0001) return?",
        ["0.01", "0.9", "0.0001", "Error - key not found"]
    )

    test = ExerciseTest("Dictionary Access", hint=".get() returns the default if key is missing")
    test.check_true(answer == 2, "Correct! 'weight_decay' doesn't exist, so default 0.0001 is returned")
    return test.run()


# =============================================================================
# SECTION 7: List Comprehensions
# =============================================================================

@lesson.section("List Comprehensions")
def section_comprehensions():
    """Learn about list comprehensions - a Pythonic way to create lists."""

    print("List comprehensions create lists in a single line:\n")

    print("# Traditional loop:")
    print("squares = []")
    print("for x in range(5):")
    print("    squares.append(x ** 2)")
    squares = []
    for x in range(5):
        squares.append(x ** 2)
    print(f"squares = {squares}")

    print("\n# List comprehension (same result, one line):")
    print("squares = [x ** 2 for x in range(5)]")
    squares = [x ** 2 for x in range(5)]
    print(f"squares = {squares}")

    print("\n# With condition:")
    print("even_squares = [x ** 2 for x in range(10) if x % 2 == 0]")
    even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
    print(f"even_squares = {even_squares}")

    print("\n# Practical example - layer sizes:")
    print("sizes = [784] + [256 // (2**i) for i in range(3)] + [10]")
    sizes = [784] + [256 // (2**i) for i in range(3)] + [10]
    print(f"sizes = {sizes}")

    print("\n" + "-"*50)
    print("KEY INSIGHT: You'll see this pattern often:")
    print("  [p.numel() for p in model.parameters()]  # Count all params")
    print("-"*50)


@lesson.exercise("List Comprehension", points=1)
def exercise_comprehensions():
    """Test understanding of list comprehensions."""

    answer = ask_question(
        "What is [x * 2 for x in [1, 2, 3]]?",
        ["[1, 2, 3]", "[2, 4, 6]", "[1, 4, 9]", "[3, 6, 9]"]
    )

    test = ExerciseTest("List Comprehension", hint="Each element x is multiplied by 2")
    test.check_true(answer == 1, "Correct! Each element is doubled: [2, 4, 6]")
    return test.run()


# =============================================================================
# FINAL CODING EXERCISE
# =============================================================================

@lesson.exercise("Coding Challenge: Build a Config", points=3)
def exercise_final():
    """Final coding exercise to test all concepts."""

    print("Let's test what you've learned!\n")
    print("Task: Create a training config dictionary with:")
    print("  1. learning_rate: 0.001")
    print("  2. epochs: 50")
    print("  3. layer_sizes: [784, 256, 128, 10]")
    print("  4. use_gpu: True")

    # Create the config
    config = {
        "learning_rate": 0.001,
        "epochs": 50,
        "layer_sizes": [784, 256, 128, 10],
        "use_gpu": True
    }

    # Test it
    test = ExerciseTest("Training Config")

    # Verify the config
    print("\nVerifying config...")
    test.check_true(config["learning_rate"] == 0.001, "learning_rate = 0.001")
    test.check_true(config["epochs"] == 50, "epochs = 50")
    test.check_true(config["layer_sizes"] == [784, 256, 128, 10], "layer_sizes correct")
    test.check_true(config["use_gpu"] == True, "use_gpu = True")

    # Bonus: Test comprehension
    print("\nBonus: Computing total neurons...")
    total_neurons = sum(config["layer_sizes"])
    test.check_true(total_neurons == 784 + 256 + 128 + 10, f"Total neurons = {total_neurons}")

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
