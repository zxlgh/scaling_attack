import os
import random

import cv2 as cv

path = r'/home/pub-60/benign/train/1'
files = os.listdir(path)
img = cv.imread(os.path.join(path, random.choice(files)))
print(img.shape)