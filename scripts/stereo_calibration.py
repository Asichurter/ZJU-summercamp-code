######################################################
# 浙江大学软件学院夏令营代码
# --------------------
# 作者:唐郅杰
# 导师: 蔡登
# --------------------
#
# 本文件中主要使用opencv进行双目摄像机的标定。先利用
# calibrateCamera逐个标定单目摄像机获得相机矩阵，然后利用两个摄像
# 机的相机矩阵计算两个相机坐标系的转换，利用APi:stereoCalibrate
######################################################

import numpy as np
import cv2
import glob

# 迭代停止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备建系在棋盘上的棋盘角点的坐标,Z坐标全部为0
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 存储棋盘点的3D世界坐标,为棋盘点坐标
left_imgpoints = [] # 2D图像坐标,利用cv2的API求得
right_imgpoints = [] # 2D图像坐标,利用cv2的API求得


left_imgs = glob.glob('../imgs/left/*.jpg')
right_imgs = glob.glob('../imgs/right/*.jpg')

print('收集图像点坐标...')
for left,right in zip(left_imgs,right_imgs):
    l_img = cv2.imread(left)
    r_img = cv2.imread(right)

    l_gray = cv2.cvtColor(l_img,cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r_img,cv2.COLOR_BGR2GRAY)

    l_img_size = l_gray.shape[::-1]     #　图像尺寸为(w,h)
    r_img_size = r_gray.shape[::-1]     #　图像尺寸为(w,h)

    # 利用cv的API找到棋盘点在图像中的2D坐标
    l_ret, l_corners = cv2.findChessboardCorners(l_gray, (7, 6), None)
    r_ret, r_corners = cv2.findChessboardCorners(r_gray, (7, 6), None)

    # 如果找到了棋盘点,将该场景下的3D和2D坐标加入到存储中
    if l_ret and r_ret:
        objpoints.append(objp)

        #--------------------------------------------------------------------------#
        # 左摄像机，寻找棋盘点的亚像素位置
        l_corners2 = cv2.cornerSubPix(l_gray, l_corners, (11, 11), (-1, -1), criteria)

        if [l_corners2]:
            left_imgpoints.append(l_corners2)
        else:
            left_imgpoints.append(l_corners)
        #--------------------------------------------------------------------------#
        # 右摄像机，寻找棋盘点的亚像素位置
        r_corners2 = cv2.cornerSubPix(r_gray, r_corners, (11, 11), (-1, -1), criteria)

        if [r_corners2]:
            right_imgpoints.append(r_corners2)
        else:
            right_imgpoints.append(r_corners)
        # --------------------------------------------------------------------------#

print('逐个标定单目摄像机...')
ret, l_mtx, l_dist, l_rvecs, l_tvecs = cv2.calibrateCamera(objpoints, left_imgpoints, l_img_size, None, None)
ret, r_mtx, r_dist, r_rvecs, r_tvecs = cv2.calibrateCamera(objpoints, right_imgpoints, r_img_size, None, None)

print('标定双目摄像机...')
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints,
                        left_imgpoints,
                        right_imgpoints,
                        l_mtx,
                        l_dist,
                        r_mtx,
                        r_dist,
                        l_img_size)

print('*'*50)
print('左摄像机到右摄像机坐标系的旋转矩阵:','\n',R)
print('左摄像机到右摄像机坐标系的位移矩阵:','\n',T)
print('左摄像机到右摄像机坐标系的本征矩阵:','\n',E)
print('左摄像机到右摄像机坐标系的基础矩阵:','\n',F)




