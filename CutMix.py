# https://arxiv.org/pdf/1905.04899.pdf
# Let x ∈ R(W×H×C) and y denote a training image and
# its label, respectively. The goal of CutMix is to generate a
# new training sample (˜x, y˜) by combining two training samples (xA, yA) and (xB, yB).

# x_bar = M*xa + (1-M)*xb  -> * = element wise 
# y_bar = (lambda)ya + (1-lambda)yb
# M belongs to {0, 1}(W*H)
# lambda sampled from Beta(alpha, alpha) 
# alpha = 1 -> np.random.beta(1, 1)
# To sample M first sample the Bounding Box coordinate (rx, ry, rw, rh)
# Bounding box indicate the cropping region in xa & xb
# rx ~ Unif (0, W)
# ry ~ Unif (0, H)
# rw = W*np.sqrt(1- lambda)
# rh = H*np.sqrt(1- lambda)

# in the cropping region the bounding box 
# is filled with zero else 
# everywhere it is 1

import numpy as np 
def rand_bbox(size):
    H = size[0]
    W = size[1]

    rx = np.random.uniform(0, W)
    ry = np.random.uniform(0, H)
    
    lam = np.random.beta(1, 1)
    rw = W*(np.sqrt(1-lam))
    rh = H*(np.sqrt(1-lam))

    return int(rx), int(ry), int(rw), int(rh), lam

def apply_cutmix(img1, img2, label1, label2):
    shape = img1.shape
    # shape -> H, W, C
    rx, ry, rw, rh, lam = rand_bbox(shape[:2])
    M = np.ones(shape = shape)

    if len(shape) ==3 :
        M[(ry - rh//2):(ry + rh//2), (rx - rw//2):(rx + rw//2), :] = 0 
    elif len(shape) == 2:
        M[(ry - rh//2):(ry + rh//2), (rx - rw//2):(rx + rw//2)] = 0 
    else :
        print("The number of color channels in the image can be 1 or 3 only")

    x_bar = np.multiply(M, img1) + np.multiply((1-M), img2)
    y_bar = lam*label1 + (1-lam)*label2

    return x_bar, y_bar
    
# example
import pandas as pd
data = pd.read_feather("sample_128-128.feather")
img1 = data.iloc[0, 1:].values.reshape(128, 128).astype(np.uint8)
img2 = data.iloc[1, 1:].values.reshape(128, 128).astype(np.uint8)
label1 = np.zeros(shape= (10))
label2 = np.zeros(shape= (10))
label1[5] = 1
label2[2] = 1
x_bar, y_bar = apply_cutmix(img1, img2, label1, label2)

import matplotlib.pyplot as plt 
plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

plt.imshow(x_bar)
plt.show()
