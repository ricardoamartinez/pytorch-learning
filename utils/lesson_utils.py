"""
Lesson Utilities - Interactive learning framework for PyTorch course.

Provides:
- Unit test framework for exercises
- Progress tracking
- Interactive prompts
- Code validation helpers
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from functools import wraps
import traceback

# =============================================================================
# COLORS AND FORMATTING
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def success(msg: str) -> None:
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def error(msg: str) -> None:
    """Print error message in red."""
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def info(msg: str) -> None:
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def warning(msg: str) -> None:
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def header(msg: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def subheader(msg: str) -> None:
    """Print subsection header."""
    print(f"\n{Colors.BOLD}{'-'*50}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'-'*50}{Colors.END}\n")

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

PROGRESS_FILE = Path(__file__).parent.parent / ".progress.json"

def load_progress() -> dict:
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_lessons": [], "completed_exercises": {}, "scores": {}}

def save_progress(progress: dict) -> None:
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def mark_exercise_complete(lesson: str, exercise: str) -> None:
    """Mark an exercise as complete."""
    progress = load_progress()
    if lesson not in progress["completed_exercises"]:
        progress["completed_exercises"][lesson] = []
    if exercise not in progress["completed_exercises"][lesson]:
        progress["completed_exercises"][lesson].append(exercise)
    save_progress(progress)

def mark_lesson_complete(lesson: str, score: int, total: int) -> None:
    """Mark a lesson as complete with score."""
    progress = load_progress()
    if lesson not in progress["completed_lessons"]:
        progress["completed_lessons"].append(lesson)
    progress["scores"][lesson] = {"score": score, "total": total}
    save_progress(progress)

def show_progress() -> None:
    """Display current progress."""
    progress = load_progress()
    header("YOUR PROGRESS")

    if not progress["completed_lessons"]:
        info("No lessons completed yet. Let's start learning!")
        return

    print(f"Completed lessons: {len(progress['completed_lessons'])}/15\n")
    for lesson in sorted(progress["completed_lessons"]):
        score_info = progress["scores"].get(lesson, {})
        score = score_info.get("score", "?")
        total = score_info.get("total", "?")
        print(f"  {Colors.GREEN}✓{Colors.END} {lesson}: {score}/{total}")

# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class TestResult:
    """Result of a single test."""
    def __init__(self, name: str, passed: bool, message: str = "", hint: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.hint = hint

class ExerciseTest:
    """
    A test case for an exercise.

    Usage:
        test = ExerciseTest("Create a 3x3 tensor of zeros")
        test.check_equal(student_tensor.shape, (3, 3), "Shape should be (3, 3)")
        test.check_true(torch.all(student_tensor == 0), "All values should be 0")
        test.run()
    """
    def __init__(self, name: str, hint: str = ""):
        self.name = name
        self.hint = hint
        self.checks: List[Tuple[Callable, str]] = []
        self.passed = True
        self.messages: List[str] = []

    def check_equal(self, actual: Any, expected: Any, message: str = "") -> 'ExerciseTest':
        """Check if actual equals expected."""
        def check():
            if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                return torch.allclose(actual, expected, atol=1e-5)
            elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
                return np.allclose(actual, expected, atol=1e-5)
            return actual == expected
        self.checks.append((check, message or f"Expected {expected}, got {actual}"))
        return self

    def check_true(self, condition: bool, message: str = "") -> 'ExerciseTest':
        """Check if condition is True."""
        self.checks.append((lambda: condition, message or "Condition should be True"))
        return self

    def check_false(self, condition: bool, message: str = "") -> 'ExerciseTest':
        """Check if condition is False."""
        self.checks.append((lambda: not condition, message or "Condition should be False"))
        return self

    def check_shape(self, tensor: torch.Tensor, expected_shape: tuple, message: str = "") -> 'ExerciseTest':
        """Check tensor shape."""
        def check():
            return tuple(tensor.shape) == expected_shape
        self.checks.append((check, message or f"Expected shape {expected_shape}, got {tuple(tensor.shape)}"))
        return self

    def check_dtype(self, tensor: torch.Tensor, expected_dtype: torch.dtype, message: str = "") -> 'ExerciseTest':
        """Check tensor dtype."""
        def check():
            return tensor.dtype == expected_dtype
        self.checks.append((check, message or f"Expected dtype {expected_dtype}, got {tensor.dtype}"))
        return self

    def check_range(self, tensor: torch.Tensor, min_val: float, max_val: float, message: str = "") -> 'ExerciseTest':
        """Check if tensor values are within range."""
        def check():
            return tensor.min() >= min_val and tensor.max() <= max_val
        self.checks.append((check, message or f"Values should be in [{min_val}, {max_val}]"))
        return self

    def check_callable(self, func: Callable, message: str = "") -> 'ExerciseTest':
        """Check if function returns True."""
        self.checks.append((func, message or "Custom check failed"))
        return self

    def run(self) -> bool:
        """Run all checks and report results."""
        all_passed = True

        for check_func, message in self.checks:
            try:
                if check_func():
                    success(message or "Check passed")
                else:
                    error(message)
                    all_passed = False
            except Exception as e:
                error(f"{message} (Error: {e})")
                all_passed = False

        if all_passed:
            success(f"Exercise '{self.name}' PASSED!")
        else:
            error(f"Exercise '{self.name}' FAILED")
            if self.hint:
                warning(f"Hint: {self.hint}")

        return all_passed

class LessonRunner:
    """
    Runs a lesson with multiple sections and exercises.

    Usage:
        lesson = LessonRunner("02_tensors")

        @lesson.section("Creating Tensors")
        def section_1():
            # Teaching content...
            pass

        @lesson.exercise("Create a zeros tensor")
        def exercise_1():
            # Return True if passed, False if failed
            return test.run()

        lesson.run()
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.sections: List[Tuple[str, Callable]] = []
        self.exercises: List[Tuple[str, Callable]] = []
        self.passed_exercises = 0
        self.total_exercises = 0

    def section(self, title: str) -> Callable:
        """Decorator to add a section."""
        def decorator(func: Callable) -> Callable:
            self.sections.append((title, func))
            return func
        return decorator

    def exercise(self, title: str, points: int = 1) -> Callable:
        """Decorator to add an exercise."""
        def decorator(func: Callable) -> Callable:
            self.exercises.append((title, func, points))
            self.total_exercises += points
            return func
        return decorator

    def run_interactive(self) -> None:
        """Run the lesson interactively."""
        header(f"LESSON: {self.name}")
        if self.description:
            print(f"{self.description}\n")

        for title, section_func in self.sections:
            subheader(title)
            try:
                section_func()
            except Exception as e:
                error(f"Error in section: {e}")
                traceback.print_exc()

            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")

        # Run exercises
        if self.exercises:
            header("EXERCISES")
            print(f"Complete {len(self.exercises)} exercise(s) to finish this lesson.\n")

            for title, exercise_func, points in self.exercises:
                subheader(f"Exercise: {title}")
                try:
                    if exercise_func():
                        self.passed_exercises += points
                        mark_exercise_complete(self.name, title)
                except Exception as e:
                    error(f"Error in exercise: {e}")
                    traceback.print_exc()

                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")

        # Show results
        self._show_results()

    def run(self) -> None:
        """Run the lesson (non-interactive mode for testing)."""
        header(f"LESSON: {self.name}")
        if self.description:
            print(f"{self.description}\n")

        for title, section_func in self.sections:
            subheader(title)
            try:
                section_func()
            except Exception as e:
                error(f"Error in section: {e}")

        # Run exercises
        if self.exercises:
            header("EXERCISES")
            for title, exercise_func, points in self.exercises:
                subheader(f"Exercise: {title}")
                try:
                    if exercise_func():
                        self.passed_exercises += points
                        mark_exercise_complete(self.name, title)
                except Exception as e:
                    error(f"Error in exercise: {e}")

        self._show_results()

    def _show_results(self) -> None:
        """Show final results."""
        header("LESSON COMPLETE")

        percentage = (self.passed_exercises / self.total_exercises * 100) if self.total_exercises > 0 else 100

        print(f"Score: {self.passed_exercises}/{self.total_exercises} ({percentage:.0f}%)\n")

        if percentage == 100:
            success("Perfect score! You've mastered this lesson!")
            mark_lesson_complete(self.name, self.passed_exercises, self.total_exercises)
        elif percentage >= 70:
            success("Good job! You passed the lesson.")
            mark_lesson_complete(self.name, self.passed_exercises, self.total_exercises)
        else:
            warning("Keep practicing! Review the sections and try the exercises again.")

        print(f"\n{Colors.CYAN}Run 'python -c \"from utils.lesson_utils import show_progress; show_progress()\"' to see your overall progress.{Colors.END}")

# =============================================================================
# INTERACTIVE HELPERS
# =============================================================================

def ask_question(question: str, options: List[str]) -> int:
    """
    Ask a multiple choice question.
    Returns the index of the selected option (0-based).
    """
    print(f"\n{Colors.BOLD}{question}{Colors.END}\n")
    for i, option in enumerate(options):
        print(f"  {i + 1}. {option}")

    while True:
        try:
            answer = input(f"\n{Colors.CYAN}Your answer (1-{len(options)}): {Colors.END}")
            choice = int(answer) - 1
            if 0 <= choice < len(options):
                return choice
            error(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            error("Please enter a valid number")

def ask_code(prompt: str) -> str:
    """Ask user to input code."""
    print(f"\n{Colors.BOLD}{prompt}{Colors.END}")
    print(f"{Colors.CYAN}(Enter your code, then press Enter twice to submit){Colors.END}\n")

    lines = []
    while True:
        line = input()
        if line == "":
            if lines and lines[-1] == "":
                break
            lines.append(line)
        else:
            lines.append(line)

    return "\n".join(lines[:-1])  # Remove trailing empty line

def run_student_code(code: str, context: dict = None) -> dict:
    """
    Execute student code and return the resulting namespace.

    Args:
        code: Student's code string
        context: Variables to make available to student code

    Returns:
        Dictionary of variables defined in student code
    """
    namespace = {"torch": torch, "nn": nn, "np": np}
    if context:
        namespace.update(context)

    try:
        exec(code, namespace)
        return {"success": True, "namespace": namespace}
    except Exception as e:
        return {"success": False, "error": str(e), "namespace": namespace}

def verify_tensor(tensor: Any,
                  expected_shape: tuple = None,
                  expected_dtype: torch.dtype = None,
                  expected_values: torch.Tensor = None,
                  value_range: tuple = None) -> Tuple[bool, str]:
    """
    Verify a tensor meets requirements.

    Returns:
        (passed, message) tuple
    """
    if not isinstance(tensor, torch.Tensor):
        return False, f"Expected torch.Tensor, got {type(tensor).__name__}"

    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        return False, f"Expected shape {expected_shape}, got {tuple(tensor.shape)}"

    if expected_dtype is not None and tensor.dtype != expected_dtype:
        return False, f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    if expected_values is not None:
        if not torch.allclose(tensor, expected_values, atol=1e-5):
            return False, f"Values don't match expected"

    if value_range is not None:
        min_val, max_val = value_range
        if tensor.min() < min_val or tensor.max() > max_val:
            return False, f"Values should be in range [{min_val}, {max_val}]"

    return True, "Tensor verified successfully"

def verify_model(model: nn.Module,
                 input_shape: tuple,
                 output_shape: tuple = None,
                 param_count: int = None) -> Tuple[bool, str]:
    """
    Verify a model meets requirements.

    Returns:
        (passed, message) tuple
    """
    if not isinstance(model, nn.Module):
        return False, f"Expected nn.Module, got {type(model).__name__}"

    # Test forward pass
    try:
        x = torch.randn(*input_shape)
        output = model(x)
    except Exception as e:
        return False, f"Forward pass failed: {e}"

    if output_shape is not None and tuple(output.shape) != output_shape:
        return False, f"Expected output shape {output_shape}, got {tuple(output.shape)}"

    if param_count is not None:
        actual_params = sum(p.numel() for p in model.parameters())
        if actual_params != param_count:
            return False, f"Expected {param_count} parameters, got {actual_params}"

    return True, "Model verified successfully"

# =============================================================================
# QUICK TEST HELPERS
# =============================================================================

def test_equal(actual, expected, name: str = "Test") -> bool:
    """Quick equality test with output."""
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        passed = torch.allclose(actual, expected, atol=1e-5)
    else:
        passed = actual == expected

    if passed:
        success(f"{name}: PASSED")
    else:
        error(f"{name}: FAILED (expected {expected}, got {actual})")
    return passed

def test_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "Shape test") -> bool:
    """Quick shape test with output."""
    actual_shape = tuple(tensor.shape)
    if actual_shape == expected_shape:
        success(f"{name}: PASSED - shape is {expected_shape}")
        return True
    else:
        error(f"{name}: FAILED - expected {expected_shape}, got {actual_shape}")
        return False

def test_true(condition: bool, name: str = "Test") -> bool:
    """Quick boolean test."""
    if condition:
        success(f"{name}: PASSED")
    else:
        error(f"{name}: FAILED")
    return condition

# =============================================================================
# MAIN - Show progress when run directly
# =============================================================================

if __name__ == "__main__":
    show_progress()
