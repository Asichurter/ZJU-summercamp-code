######################################################
# 浙江大学软件学院夏令营代码
# --------------------
# 作者:唐郅杰
# 导师: 蔡登
# 时间: 2020.7
# --------------------
#
# 本文件中主要使用opencv进行视差disparity的计算，主要使用了SGBM
# 算法
######################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt

img_idx = 0

imgL = cv2.imread(f'../imgs/rectified/left/{img_idx}.jpg')
imgR = cv2.imread(f'../imgs/rectified/right/{img_idx}.jpg')

# 视差范围的调整
window_size = 3
min_disp = 0
num_disp = 320 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=3,
    P1=8 * 3 * window_size ** 2,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
# disparity = disparity.transpose()
plt.axis('off')
plt.imshow(disparity, 'gray')
plt.show()