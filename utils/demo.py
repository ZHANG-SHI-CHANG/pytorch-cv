import torch
from torch import nn
import mxnet as mx
import cv2
import numpy as np

img_path = '../scripts/png/cat.jpg'

img = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2RGB)
print(isinstance(img, np.ndarray))

img_2 = mx.image.imread(img_path, 1)

print(img[0, 0, :])
print(img_2[0, 0, :])