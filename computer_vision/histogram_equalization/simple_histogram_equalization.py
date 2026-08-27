"""
================================================================================
HISTOGRAM EQUALIZATION ALGORITHM
OpenCV Function: cv2.equalizeHist(img_gray)
================================================================================

1. MECHANISM:
   - PDF (Probability Density Function): Counts the number of pixels at each intensity level (0-255).
   - CDF (Cumulative Distribution Function): Computes the cumulative sum of pixels from intensity 0 to 255.
   - Stretching: Normalizes the CDF curve to span the entire intensity range (0-255).
   - Mapping: Replaces the original intensity value of each pixel with the newly calculated intensity.

2. MAPPING FORMULA:
   h(v) = round( ((CDF(v) - CDF_min) / (M * N - CDF_min)) * 255 )

3. VARIABLES EXPLAINED:
   - v         : Original pixel intensity value (0 - 255).
   - h(v)      : New pixel intensity value after equalization.
   - CDF(v)    : Cumulative number of pixels with an intensity less than or equal to v.
   - CDF_min   : Minimum non-zero cumulative value in the image.
   - M * N     : Total number of pixels in the image (Width x Height).
   - 255       : Maximum intensity value for an 8-bit image, used to scale the range.

=> Purpose: Automatically stretches pixels concentrated in a narrow intensity range 
            (e.g., underexposed or overexposed images) across the full 0-255 range 
            to maximize global contrast.
================================================================================
"""

import cv2
import matplotlib.pyplot as plt

image_files = ['Book1.png', 'Book2.png', 'Book3.png']
titles = ['Underexposed', 'Normal', 'Overexposed']
fig, axes = plt.subplots(3, 2, figsize=(12, 18))

for i, file_name in enumerate(image_files):
    img_bgr = cv2.imread(file_name, 1)
    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    
    # 2. change BGR to YUV (so that we can extract saturation)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    img_rgb_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    axes[i, 0].imshow(img_rgb_original)
    axes[i, 0].set_title(f'Org - {titles[i]}')
    axes[i, 0].axis('off')
    
    # Vẽ ảnh đã cân bằng lên cột 2
    axes[i, 1].imshow(img_rgb_equalized)
    axes[i, 1].set_title(f'Equalized - {titles[i]}')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()