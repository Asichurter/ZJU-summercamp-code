######################################################
# 浙江大学软件学院夏令营代码
# --------------------
# 作者:唐郅杰
# 导师: 蔡登
# 时间: 2020.7
# --------------------
#
# 本文件中主要使用opencv进行左右相机图片的整形(rectification)。
# 主要是利用双目标定中获得的一系列旋转矩阵，位移矩阵和基础矩阵等数据，
# 利用opencv的stereoRectify接口完成整形，获得投影矩阵，然后利用
# initUndistortRectifyMap和remap接口完成图像整形的重映射
######################################################

import numpy as np
import cv2
import glob
import os

square_size = 60.

# 迭代停止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


# 准备建系在棋盘上的棋盘角点的坐标,Z坐标全部为0
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 存储棋盘点的3D世界坐标,为棋盘点坐标
left_imgpoints = [] # 2D图像坐标,利用cv2的API求得
right_imgpoints = [] # 2D图像坐标,利用cv2的API求得
left_valid_gray = []
right_valid_gray = []

left_img_gray_cache = None
right_img_gray_cache = None
cached_index = 1

os.system('rm -rf ../imgs/rectified/*')
os.mkdir('../imgs/rectified/right')
os.mkdir('../imgs/rectified/left')

left_imgs = glob.glob('../imgs/left/*.jpg')
right_imgs = glob.glob('../imgs/right/*.jpg')

left_imgs = sorted(left_imgs)
right_imgs = sorted(right_imgs)

excluded_idx = [12]
valid_idx = []

print('收集图像点坐标...')
for i,(left,right) in enumerate(zip(left_imgs,right_imgs)):
    if i in excluded_idx:
        continue

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
    # print(l_ret,r_ret)
    if l_ret and r_ret:
        left_valid_gray.append(l_gray)
        right_valid_gray.append(r_gray)
        valid_idx.append(i)
        objpoints.append(objp)

        #-------------------------------------opencv进行双目摄像机的标定。先利用
# calibrateCamera逐个标定单目摄像机获得相机矩阵，然后利用两个摄像
# 机的相机矩阵计算两个相机坐标系的转换，利用APi:stereoCalibrate-------------------------------------#
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
        # l_img = cv2.drawChessboardCorners(l_img, (7,6), l_corners2, l_ret)
        # r_img = cv2.drawChessboardCorners(r_img, (7,6), r_corners2, r_ret)
        # # cv2.imshow('img_left',l_img)
        # cv2.imshow('img_right', r_img)
        # cv2.waitKey(1000)

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

print('校正双目摄像机...')
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(l_mtx, l_dist,
                                                  r_mtx, r_dist,
                                                  l_img_size,
                                                  R, T)

print('初始化整形映射...')
mapx1, mapy1 = cv2.initUndistortRectifyMap(l_mtx, l_dist, R1, P1, l_img_size, cv2.CV_16SC2)
mapx2, mapy2 = cv2.initUndistortRectifyMap(r_mtx, r_dist, R2, P2, r_img_size, cv2.CV_16SC2)

# 选择一组cache的图像pair展现rectification以后的效果
for i,(l_gray,r_gray) in enumerate(zip(left_valid_gray, right_valid_gray)):

    l_rectified = cv2.remap(l_gray, mapx1, mapy1, cv2.INTER_LINEAR)
    r_rectified = cv2.remap(r_gray, mapx2, mapy2, cv2.INTER_LINEAR)

    cv2.imwrite(f'../imgs/rectified/left/{i}.jpg', l_rectified)
    cv2.imwrite(f'../imgs/rectified/right/{i}.jpg', r_rectified)

print('双目摄像机标定RMS:', retval)

# -------------------------------------------------------------------------------
# l_epi_img = cv2.imread(f'../imgs/rectified/left/{cached_index}.jpg')
# r_epi_img = cv2.imread(f'../imgs/rectified/right/{cached_index}.jpg')
#
# l_epi_gray = cv2.cvtColor(l_epi_img,cv2.COLOR_BGR2GRAY)
# r_epi_gray = cv2.cvtColor(r_epi_img,cv2.COLOR_BGR2GRAY)
#
# l_ret, l_corners = cv2.findChessboardCorners(l_epi_gray, (7, 6), None)  # True, left_imgpoints[valid_idx[cached_index]]#
# r_ret, r_corners = cv2.findChessboardCorners(r_epi_gray, (7, 6), None)  # True, right_imgpoints[valid_idx[cached_index]]#
#
# # l_img = cv2.imread('../imgs/rectified/before_left.jpg')
# # r_img = cv2.imread('../imgs/rectified/before_right.jpg')
# # l_img = cv2.drawChessboardCorners(l_img, (7,6), l_corners, l_ret)
# # r_img = cv2.drawChessboardCorners(r_img, (7,6), r_corners, r_ret)
# #
# # cv2.imwrite('../imgs/rectified/line_chessboard_left.jpg',l_img)
# # cv2.imwrite('../imgs/rectified/line_chessboard_right.jpg',r_img)
#
# l_epiline = cv2.computeCorrespondEpilines(l_corners[::7],1,F)
# r_epiline = cv2.computeCorrespondEpilines(r_corners[::7],2,F)
#
# width = l_rectified.shape[1]
# for l_line,r_line in zip(l_epiline,r_epiline):
#     l_line, r_line = l_line[0], r_line[0]
#     cv2.line(r_epi_img,(0,-l_line[2]/l_line[1]),(width,int(-(l_line[2]+l_line[0]*width)/l_line[1])), color=(0,0,255))
#     cv2.line(l_epi_img,(0,-r_line[2]/r_line[1]),(width,int(-(r_line[2]+r_line[0]*width)/r_line[1])), color=(0,0,255))
#
# cv2.imwrite('../imgs/rectified/line_left.jpg', l_epi_img)
# cv2.imwrite('../imgs/rectified/line_right.jpg', r_epi_img)
# -------------------------------------------------------------------------------

# np.save('../data/left_camera_mtx.npy', l_mtx)
# np.save('../data/right_camera_mtx.npy', r_mtx)
# np.save('../data/left_camera_proj_mtx.npy', P1)
# np.save('../data/right_camera_proj_mtx.npy', P2)


