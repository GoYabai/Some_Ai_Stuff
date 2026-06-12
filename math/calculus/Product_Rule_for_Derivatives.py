import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    f_der = [i * f_coeffs[i] for i in range(1, len(f_coeffs))]
    if not f_der:
        f_der = [0.0]
    g_der = [i * g_coeffs[i] for i in range(1, len(g_coeffs))]
    if not g_der:
        g_der = [0.0]
    a = np.convolve(f_der, g_coeffs).tolist()
    b = np.convolve(f_coeffs, g_der).tolist()
    max_len = max(len(a), len(b))
    a += [0.0] * (max_len - len(a))
    b += [0.0] * (max_len - len(b))
    c = [round(a[i] + b[i], 4) for i in range(max_len)]
    while len(c) > 1 and c[-1] == 0.0:
        c.pop()
    return c
    