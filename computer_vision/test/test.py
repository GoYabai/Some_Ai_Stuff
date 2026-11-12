import cv2
import imutils

image = cv2.imread("thoi.jpg")

(h, w, d) = image.shape
print(f"width={w}, height={h}, depth={d}")

(B, G, R) = image[100, 50]
print(f"R={R}, G={G}, B={B}")

y_min = int(h * 0.12)
y_max = int(h * 0.36)
x_min = int(w * 0.28)
x_max = int(w * 0.68)

# Cắt vùng ROI khuôn mặt mèo
cat_face_roi = image[y_min:y_max, x_min:x_max]

resized = cv2.resize(image, (200, 200))


# CV2
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45.0, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))


# Imutils
rotated = imutils.rotate(image, -45)


rotated = imutils.rotate_bound(image, 45)


blurred = cv2.GaussianBlur(image, (3, 3), 0)

output = image.copy()
cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)


output = image.copy()
cv2.putText(output, "CON MEO", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)



