import numpy as np

x = np.random.randn(4,1)
x[-1] = 1   # 3D世界坐标系中任意一点，齐次坐标形式

lmtx = np.load('../data/left_camera_mtx.npy')
rmtx = np.load('../data/right_camera_mtx.npy')
P1 = np.load('../data/left_camera_proj_mtx.npy')
P2 = np.load('../data/right_camera_proj_mtx.npy')

# 计算两个相机坐标系的外参数矩阵
T1 = np.matmul(np.linalg.inv(lmtx),P1)
T2 = np.matmul(np.linalg.inv(rmtx),P2)

x1 = np.matmul(T1, x)
x2 = np.matmul(T2, x)

baseline = x1[0]-x2[0]
print('x1',x1[0])
print('x2',x2[0])
print('b:',baseline)

