import cv2
import numpy as np

# 500 x 250
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')
img3 = cv2.imread('mainlogo.png')

rows, cols, channels = img3.shape
roi = img1[0:rows, 0:cols]

# Convert python logo to gray
img2gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

# cv2.imshow('mask', mask)

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img3_fg = cv2.bitwise_and(img3, img3, mask=mask)

dst = cv2.add(img1_bg, img3_fg)
img1[0:rows, 0:cols] = dst
cv2.imshow('res', img1)
# cv2.imshow('mask_inv', mask_inv)
# cv2.imshow('img1_bg', img1_bg)
# cv2.imshow('img3_fg', img3_fg)
cv2.imshow('dst', dst)

# add = img1+img2
# example: (155,211,79) + (50, 170, 200) = 205, 381, 279...translated to (205, 255,255).
# add = cv2.add(img1,img2)

# weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

# cv2.imshow('weighted', weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()
