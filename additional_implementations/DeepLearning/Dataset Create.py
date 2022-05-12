import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io


mat = scipy.io.loadmat('Brain.mat')
img = mat['T1']
lab = mat['label']

def normalize_img(A):
    max_val = np.max(A)
    A = A / max_val
    A*=255
    return cv2.resize(np.uint8(A),(224,224))

def normalize_label(A):
    return cv2.resize(np.uint8(A),(224,224))

for i in range(0,10):
    temp = img[:,:,i]
    temp1 = lab[:,:,i]
    img_1 = normalize_img(temp)
    label = normalize_label(temp1)
    # cv2.imshow("A",img)
    # cv2.imshow("B",label)
    img_name = 'img/' + str(i) + ".png"
    label_name = 'label/' + str(i) + ".png"
    print(img_1.shape)
    # cv2.imwrite(img_name, img_1)
    # cv2.imwrite(label_name, label)
    # cv2.waitKey()
