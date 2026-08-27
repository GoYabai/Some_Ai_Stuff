import cv2
import numpy as np
from scipy.ndimage import generic_filter

img = cv2.imread('img.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)


x = np.astype(gray, np.float32)
x_filt = generic_filter(x, np.std, size=7)
cv2.imwrite('edge_s1.jpg', x_filt)

x_filt = cv2.normalize(
    x_filt,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)
x_filt = x_filt.astype(np.uint8)
cv2.imwrite('edge_s2.jpg', x_filt)

x_filt[x_filt < 20] = 0
cv2.imwrite('edge_s3.jpg', x_filt)

x_filt = x_filt * 2.5
cv2.imwrite('edge_s4.jpg', x_filt)