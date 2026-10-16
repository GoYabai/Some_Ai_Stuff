import numpy as np
def color_to_grayscale(image: list) -> list:
    """
    Returns the luminance value of every RGB pixel.
    """
    # Write code here
    image = np.array(image)
    gray_scale = np.array([0.299, 0.587, 0.114])
    return np.sum(image * gray_scale, axis=2).tolist()
    