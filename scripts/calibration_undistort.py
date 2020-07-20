######################################################
# 浙江大学软件学院夏令营代码
# --------------------
# 作者:唐郅杰
# 导师: 蔡登
# --------------------
#
# 本文件中主要使用opencv在多个场景下,利用棋盘图进行相机标定,求出
# 相机的内部参数(包括畸变系数)和外部系数,并利用畸变系数对图像进行
# 畸变校正
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
imgpoints = [] # 2D图像坐标,利用cv2的API求得

images = glob.glob('../imgs/left/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]     #　图像尺寸为(w,h)

    # 利用cv的API找到棋盘点在图像中的2D坐标
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # 如果找到了棋盘点,将该场景下的3D和2D坐标加入到存储中
    if ret:
        objpoints.append(objp)

        # 寻找棋盘点的亚像素位置
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        if [corners2]:
            imgpoints.append(corners2)
        else:
            imgpoints.append(corners)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(1000)
        # break

# cv2.destroyAllWindows()

# 相机标定
# 返回值分别为： 标定返回值，相机矩阵，畸变系数，旋转向量和位移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 精化照相机矩阵
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,img_size,1,img_size)

# 利用相机矩阵进行畸变校正
img = cv2.imread('../imgs/left/left01.jpg')
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 裁剪图像
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('../imgs/undistort/after_undistort.jpg',dst)
# cv2.imshow("img", dst)
# cv2.waitKey(5000)
cv2.destroyAllWindows()

print('相机矩阵:', mtx)
print('畸变系数:', dist)
print('旋转向量:', rvecs)
print('位移向量:', tvecs)