import cv2
import numpy as np

# add noise
image = cv2.imread('mrbean.jpg')
    
row,col,ch = image.shape
s_vs_p = 0.5
amount = 0.009
out = np.copy(image)
# Salt mode
num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
      for i in image.shape]
out[tuple(coords)] = 255

# Pepper mode
num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
out[tuple(coords)] = 0


# cv2.imwrite('mrbean_noise.jpg', out)

# Noise reduction
noisy_image = cv2.imread('mrbean_noise.jpg')

# kernel size
K = 3

# noise reduction using sliding window to cal the median and then replace the noise pixel with it
cleaned_image = cv2.medianBlur(noisy_image, K)


comparison = np.concatenate((noisy_image, cleaned_image), axis=1)
cv2.imshow('Left: Noisy Image | Right: Cleaned by Median', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()