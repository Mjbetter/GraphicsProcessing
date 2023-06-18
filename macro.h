#pragma once

/*自定义π值*/
#define PAI 3.141592653589793

/*不进行滤波处理*/
#define NO_FILTER 0
/*均值滤波*/
#define AVERAGE_FILTER 1
/*中值滤波*/
#define MEDIAN_FILTER 2
/*高斯滤波*/
#define GAUSSIAN_FILTER 3
/*双边滤波*/
#define BILATERAL_FILTER 4
/*小波滤波*/
#define SMALLWAVE_FILTER 5

/*边缘检测*/
/*roberts边缘检测*/
#define ROBERTS 1
/*sobel边缘检测*/
#define SOBEL 2
/*prewitt边缘检测*/
#define PREWITT 3
/*kirsch边缘检测*/
#define KIRSCH 4
/*robinson边缘检测*/
#define ROBINSON 5
/*laplacian边缘检测*/
#define LAPLACIAN 6
/*canny边缘检测*/
#define CANNY 7

/*图像增强*/
/*对比度增强*/
#define CONTRAST_ENHANCE 1
/*亮度增强*/
#define BRIGHTNESS 2
/*直方图均衡化*/
#define HISTOGRAME_QUALIZATION 3
/*指数变换增强*/
#define EXPONENTIAL_TRANSFORM 4


/*图像的四种基本变换*/
/*图像平移*/
#define IMAGE_TRANSLATION 1
/*图像缩放*/
#define IMAGE_RESIZING 2
/*图像旋转*/
#define IMAGE_ROTATING 3
/*图像镜像*/
#define IMAGE_REFLECTION 4

/*彩色图像变灰度图像*/
/*图像变为灰度图像*/
#define IMAGE_GRAYSCALE 1
/*图像变为2值图像*/
#define IMAGE_GRAYBINARY 2

/*图像的加噪*/
/*加高斯噪声*/
#define GAUSSIANNOISE 1
/*加椒盐噪声*/
#define SALTPEPPERNOISE 2
/*加泊松噪声*/
#define POISSONNOISE 3