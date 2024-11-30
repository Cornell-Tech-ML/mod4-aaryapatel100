"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The result of multiplying `x` and `y`.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (Any): The input value.

    Returns:
    -------
        Any: The same input value.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The result of adding `x` and `y`.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The negated value of `x`.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` is less than `y`, `False` otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` is equal to `y`, `False` otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of `x` and `y`.

    """
    if x > y:
        return x
    return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` and `y` are close within a tolerance of 0.01, `False` otherwise.

    """
    return math.fabs(x - y) < (1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of `x`.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: `x` if `x` is positive, `0` otherwise.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of `x`.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of `x`.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The reciprocal of `x`.

    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x (float): The input value.
        y (float): The second argument, typically a gradient.

    Returns:
    -------
        float: The derivative of log(x) multiplied by `y`.

    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        x (float): The input value.
        y (float): The second argument, typically a gradient.

    Returns:
    -------
        float: The derivative of 1/x multiplied by `y`.

    """
    return -y / (x**2)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x (float): The input value.
        y (float): The second argument, typically a gradient.

    Returns:
    -------
        float: The derivative of ReLU(x) multiplied by `y`.

    """
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float and returns a float.
        ls (Iterable[float]): An iterable of floats.

    Returns:
    -------
        Iterable[float]: A new iterable where each element is the result of applying `fn` to the corresponding element in `ls`.

    """
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float],
    ls_one: Iterable[float],
    ls_two: Iterable[float],
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two floats and returns a float.
        ls_one (Iterable[float]): The first iterable of floats.
        ls_two (Iterable[float]): The second iterable of floats.

    Returns:
    -------
        Iterable[float]: A new iterable where each element is the result of applying `fn` to the corresponding elements of `ls_one` and `ls_two`.

    """
    return [fn(x, y) for x, y in zip(ls_one, ls_two)]


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two floats and returns a float.
        ls (Iterable[float]): An iterable of floats.

    Returns:
    -------
        float: A single value resulting from the cumulative application of `fn` to the elements of `ls`.

    """
    if ls:
        it = iter(ls)
        result = next(it)
        for e in it:
            result = fn(result, e)
        return result
    return 0


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map.

    Args:
    ----
        a (Iterable[float]): A list of floats.

    Returns:
    -------
        Iterable[float]: A new list where each element is the negation of the corresponding element in `a`.

    """
    return map(neg, a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        a (Iterable[float]): The first list of floats.
        b (Iterable[float]): The second list of floats.

    Returns:
    -------
        Iterable[float]: A new list where each element is the sum of the corresponding elements in `a` and `b`.

    """
    return zipWith(add, a, b)


def sum(a: Iterable[float]) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        a (Iterable[float]): A list of floats.

    Returns:
    -------
        float: The sum of all elements in `a`.

    """
    return reduce(add, a)


def prod(a: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
    ----
        a (Iterable[float]): A list of floats.

    Returns:
    -------
        float: The product of all elements in `a`.

    """
    return reduce(mul, a)
