import math


def factorial(x):
    """
    Calculate the factorial of x (x! where x is a non-negative integer).
    """
    return math.factorial(x)


def approximate_sin(x, n):
    """
    Approximate sin(x) using a Taylor series with n+1 terms.

    Args:
        x (float): angle in radians.
        n (int): positive integer, number of iterations.

    Returns:
        sin_approx (float): the approximate value of sin(x).
    """
    r = 0
    for i in range(n + 1):
        r += ((-1)**i * x**(2 * i + 1)) / factorial(2 * i + 1)
    return r
    


def approximate_cos(x, n):
    """
    Approximate cos(x) using a Taylor series with n+1 terms.

    Args:
        x (float): angle in radians.
        n (int): positive integer, number of iterations.

    Returns:
        cos_approx (float): the approximate value of cos(x).
    """
    r = 0
    for i in range(n + 1):
        r += ((-1)**i * x**(2 * i)) / factorial(2 * i)
    return r

def approximate_sinh(x, n):
    """
    Approximate sinh(x) using a Taylor series with n+1 terms.

    Args:
        x (float): the input value.
        n (int): positive integer, number of iterations.

    Returns:
        sinh_approx (float): the approximate value of sinh(x).
    """
    r = 0
    for i in range(n + 1):
        r += (x**(2 * i + 1)) / factorial(2 * i + 1)
    return r
    


def approximate_cosh(x: float, n: int) -> float:
    """
    Approximate cosh(x) using a Taylor series with n+1 terms.

    Args:
        x (float): the input value.
        n (int): positive integer, number of iterations.

    Returns:
        cosh_approx (float): the approximate value of cosh(x).
    """
    r = 0
    for i in range(n + 1):
        r += (x**(2 * i)) / factorial(2 * i)
    return r



if __name__ == "__main__":
    assert factorial(0) == 1
    assert factorial(4) == 24
    assert round(approximate_sin(x=1, n=10), 4) == round(math.sin(1), 4)
    assert round(approximate_cos(x=1, n=10), 2) == 0.54
    assert round(approximate_sinh(x=1, n=10), 2) == 1.18
    assert round(approximate_cosh(x=1, n=10), 2) == 1.54
    print("All tests passed.")
    print(f'sin(3.14) ~ {round(approximate_sin(x=3.14, n=10), 4)}')
    print(f'cos(3.14) ~ {round(approximate_cos(x=3.14, n=10), 2)}')
    print(f'sinh(3.14) ~ {round(approximate_sinh(x=3.14, n=10), 2)}')
    print(f'cosh(3.14) ~ {round(approximate_cosh(x=3.14, n=10), 2)}')
