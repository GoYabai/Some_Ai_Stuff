import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    g_der = [g_coeffs[i] * (len(g_coeffs) - 1 - i) for i in range(len(g_coeffs) - 1)]
    if not g_der: 
        g_der = [0.0]
        
    h_der = [h_coeffs[i] * (len(h_coeffs) - 1 - i) for i in range(len(h_coeffs) - 1)]
    if not h_der: 
        h_der = [0.0]
    g_val = np.polyval(g_coeffs, x)
    h_val = np.polyval(h_coeffs, x)
    g_der_val = np.polyval(g_der, x)
    h_der_val = np.polyval(h_der, x)
    derivative_value = (g_der_val * h_val - g_val * h_der_val) / (h_val ** 2)
    return derivative_value