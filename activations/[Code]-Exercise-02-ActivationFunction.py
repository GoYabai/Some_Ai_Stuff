import math


def is_number(n):
    """
    Check whether n is a valid number
    by trying to cast it to float.
    Returns True if successful, False otherwise.
    """
    try:
        float(n)
    except ValueError:
        return False
    return True


def calculate_relu(x):
    """
    Calculate the ReLU function:
        ReLU(x) = max(0, x)

    Args:
        x (float): the input value.

    Returns:
        result (float): the value of ReLU(x).
    """
    return float(max(0, x))
    


def calculate_sigmoid(x):
    """
    Calculate the Sigmoid function:
        sigmoid(x) = 1 / (1 + e^(-x))

    Uses math.exp() instead of math.e ** (-x) for better numerical stability.

    Args:
        x (float): the input value.

    Returns:
        float: the value of sigmoid(x).
    """
    return float(1 / (1 + math.exp(-x)))


def calculate_elu(x):
    """
    Calculate the ELU (Exponential Linear Unit) function:
        ELU(x) = x                  if x >= 0
               = alpha * (e^x - 1)  if x < 0

    Note: this exercise uses alpha = 0.01 as required by the problem.

    Args:
        x (float): the input value.

    Returns:
        result (float): the value of ELU(x).
    """
    alpha = 0.01
    if x >= 0:
        return x
    else:
        return alpha * (math.exp(x) - 1)
    


def calculate_activation_function(x, act_name):
    """
    Calculate the activation function for x based on act_name.

    Args:
        x (float): the input value.
        act_name (str): name of the activation function, one of
            'relu', 'sigmoid', or 'elu'.

    Returns:
        float: the value of f(x) corresponding to act_name,
        or None if act_name is invalid.
    """
    if act_name == "relu":
        return calculate_relu(x)
    elif act_name == "sigmoid":
        return calculate_sigmoid(x)
    elif act_name == "elu":
        return calculate_elu(x)
    else:
        return None
    


def interactive_activation_function():
    while True:
        act = input("What function do you want to use (relu, sigmoid, elu): ")
        if act in ["relu", "sigmoid", "elu"]:
            break
    x = float(input("Enter x: "))
    print(calculate_activation_function(x, act))


if __name__ == "__main__":
    # Quick tests
    assert is_number(3) is True
    assert is_number('-2a') is False
    assert calculate_relu(-3) == 0.0
    assert calculate_relu(5) == 5.0
    assert round(calculate_sigmoid(3), 2) == 0.95
    assert round(calculate_elu(1)) == 1
    assert calculate_activation_function(x=1, act_name='relu') == 1.0
    print("All tests passed.")

    # Interactive run
    interactive_activation_function()
