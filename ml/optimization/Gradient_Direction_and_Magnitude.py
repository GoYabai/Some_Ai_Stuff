import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    # Your code here
    z = 0
    for num in gradient:
        z += num**2
    z = z**0.5
    
    if z == 0:
        zeros_vector = [0.0] * len(gradient)
        return {
            "magnitude": 0.0,
            "direction": zeros_vector,
            "descent_direction": zeros_vector
        }
        
    d = {
        "magnitude": z,
        "direction": [num / z for num in gradient],
        "descent_direction": [-num / z for num in gradient]
    }
    
    return d