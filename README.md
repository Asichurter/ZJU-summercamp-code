# ZJU-summercamp-code
本项目代码是浙江大学软件学院2020年暑期夏令营AZFT分营的项目代码

## 基本信息
- **选择导师**：蔡登
- **项目内容**:  Stereo

- **主要学习资料**：Opencv官方文档，Opencv官方论坛，Stackoverflow论坛，CSDN论坛，博客园，谷歌学术等等

- **运行环境配置**：
  - Ubuntu 18.04 LTS
  - Python 3.7
  - Opencv-python 4.3.0.36
  - PyCharm Professional 2019.3.4

- **Python库依赖**：
  - Numpy
  - glob
  - opencv-python



## 项目说明

本项目主要是立体视觉项目的代码。

### 代码

均位于scripts文件夹中

- **calibration_undistort.py**:

  实现单目相机的标定calibration，并且利用标定获得的畸变系数将图像对图像进行畸变校正(undistort)。校正的图像源路径可以修改undistort_img_path变量，校正后的保存位置可以修改undistort_dst_path变量

  

  运行（根目录下）：

  ```
  python scripts/calibration_undistort.py
  ```

  

- **stereo_calibration.py**

  实现了双目相机的标定，计算旋转矩阵$R$，位移向量$t$，基础矩阵$F$等

  

  运行(根目录下):	

  ```
  python scripts/calibration_undistort.py
  ```



- **rectification.py**

  实现了双目相机的整型rectify，使得两摄像机的光轴平行共向。整型将会使用双目标定获得的旋转矩阵$R$和位移向量$t$，生成基础矩阵，两个摄像机分别的整型旋转矩阵*R1*,*R2*​和整型位移向量​*t1*,*t2*​,并生成最终用于整型图像的映射。

  校正后的图像将会保存在 [imgs/rectified/left]() 和 [imgs/rectified/left]() 中

  

  运行(根目录下):	

  ```
  python scripts/rectification.py
  ```



- **draw_epipolar_line.py**

  画出双目摄像机整型rectified后，左右对应图像的极线epipolar line

  

  可以通过修改 used_idx 变量修改想要画极线的图像（0-4之间），画出的极线保存在 [imgs/rectified/line_left.jpg]() 和  [imgs/rectified/line_right.jpg]() 中

  

  运行(根目录下):	

  ```
  python scripts/draw_epipolar_line.py
  ```



- **cal_baseline.py**

  利用双目摄像机整型后得到的投影矩阵*P1*,*P2*和两个摄像机的相机矩阵*K1*,*K2*，计算摄像机整型后的基线baseline:

  使用的投影矩阵和相机矩阵的数据来自rectification.py运行后的保存数据，数据位于 [data/]() 中

  

  运行(根目录下):	

  ```
  python scripts/cal_baseline.py
  ```



- **SGBM.py**

  利用整型后的图像，计算图像中对应点的视深 disparity

  

  运行(根目录下):	

  ```
  python scripts/SGBM.py
  ```



## 图像

主要位于 [imgs/]() 文件夹中，其中：

-  [imgs/left]() 和 [imgs/right]() 中包含最原生的左右两摄像机的照片
- [imgs/rectified]() 包含整型rectified后的图像和画出极线的整形后图像
- [imgs/SGBM]() 包含使用SGBM算法估计视深后的视深灰度图



## 数据

主要位于 [data]() 文件夹下，包含整型rectified后的相机矩阵数据*K1*,*K2*​和投影矩阵数据*P1*,*P2*​