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
/*小边滤波*/
#define SMALLWAVE_FILTER 5

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

/*加椒盐噪声*/

/*加泊松噪声*/
