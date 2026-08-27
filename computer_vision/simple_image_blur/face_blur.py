import cv2
import numpy as np

# install frontal face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# read img
image = cv2.imread('mrbean.jpg', 1)

# change img to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# face detect
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# iter all the detected face
for (x, y, w, h) in faces:
    # draw green box
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # face
    roi = image[y:y+h, x:x+w]

    # kernel size
    K = 13
    kernel = np.ones((K, K), np.float32) / (K*K)
    
    #blur the face
    roi_blurred = cv2.filter2D(roi, -1, kernel)

    # asign back to the org img
    image[y:y+h, x:x+w] = roi_blurred

# show img
cv2.imshow('Mean Filter Practice', image)
cv2.waitKey(0)
cv2.destroyAllWindows()