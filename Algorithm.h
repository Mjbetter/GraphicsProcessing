#pragma once
#include <opencv2\opencv.hpp>//加载OpenCV 4.0头文件
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include<opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "macro.h"
#include <string.h>

using namespace cv;
using namespace std;
using namespace cv::ml;

//**自定义结构体
struct MyNum
{
	cv::Mat mat; //数字图片
	cv::Rect rect;//相对整张图所在矩形
	int label;//数字标签
};

class ImageAlgorithm
{
public:

	/*构造积分图*/
	void GetIntegralImage(Mat img, double **integralImage);
	/*扩充图像边缘*/
	Mat imageEdgeExpand(Mat img, int size);


	/*基础功能*/

	/*图像加载*/
	Mat imageLoading_Show(string imageName);

	/*图像的四种基本变换*/
	Mat imageNoiseAddition(Mat img, int dx, int dy, double Scale_x, double Scale_y, double angle, int choice, int option);
	/*图像平移*/
	Mat imageTranslation(Mat img, int dx, int dy);
	/*图像缩放*/
	Mat imageResizing(Mat img, double Scale_x, double Scale_y);
	/*图像旋转*/
	Mat imageRotating(Mat img/*, double img_cols, double img_rows*/, double angle);
	/*图像镜像*/
	Mat imageReflection(Mat img, int choice);

	/*图像变灰度*/
	Mat imageGray(Mat img, int option);
	/*灰度图像*/
	Mat imageGrayScale(Mat img);
	/*2值图像*/
	Mat imageGrayBinary(Mat img);

	/*图像钝化*/
	Mat imageBlurring(Mat img);
	/*图像锐化*/
	Mat imageSharpening(Mat img);

	/*图像的加噪*/
	Mat imageAddNoise(Mat img, int option);
	/*高斯噪声*/
	Mat imageGaussianNoise(Mat img);
	/*椒盐噪声*/
	Mat imageSaltPepperNoise(Mat img);
	/*泊松噪声*/
	Mat imagePoissonNoise(Mat img);

	/*图像的直方图绘制*/
	Mat imageHistogram(Mat img);

	/*图像的去噪*/
	Mat imageDenoising(Mat img, int kernel_size=3, int option=AVERAGE_FILTER,int level=3);
	/*均值滤波*/
	Mat imageAverageFilter(Mat img, int kernel_size);
	/*中值滤波*/
	Mat imageMedianFilter(Mat img, int kernel_size);
	/*高斯滤波*/
	Mat imageGaussianFilter(Mat img, int kernel_size);
	/*双边滤波*/
	Mat imageBilateralFilter(Mat img, int kernel_size,double space_sigma=3,double color_sigma=3.5);
	/*小波变换*/
	void waveLetTransform(double** data, double** lowPass, double** highPass,int rows, int cols);
	/*求阈值*/
	void getThreashold(double** data,double *threashold,int rows, int cols);
	/*阈值处理*/
	void doThreashold(double** data, double *threadshold, int rows, int cols);
	/*小波反变换*/
	void inverseWaveLetTransform(double** data, double** lowPass, double** highPass, int rows, int cols);
	/*小波滤波*/
	Mat imageWaveletFilter(Mat img, int level);

	/*中级功能*/

	/*图像的边缘提取*/
	Mat imageEdgeDetection(Mat img,int option,int threshold=20);
	/*一阶边缘检测算子Roberts*/
	Mat imageRoberts(Mat img, int threshold=20);
	/*一阶边缘检测算子Sobel*/
	Mat imageSobel(Mat img,int kernel_size=3);
	/*一阶边缘检测算子Prewitt*/
	Mat imagePrewitt(Mat img);
	/*一阶边缘检测算子Kirsch*/
	Mat imageKirsch(Mat img);
	/*一阶边缘检测算子Robinson*/
	Mat imageRobinson(Mat img);
	/*一阶边缘检测算子Canny*/
	Mat imageCanny(Mat img);
	/*二阶边缘检测算子Laplacian算子*/
	Mat imageLaplacian(Mat img);


	/*图像的增强*/
	Mat imageEnhance(Mat img,int option,double c = 1.1, double r = 1.05);
	/*对比度增强*/
	Mat imageContrastEnhance(Mat img, double L);
	/*亮度增强*/
	Mat imageBrightness(Mat img, double L);
	/*统计直方图*/
	void imgageStatisticalHistogram(Mat img, int **hist);
	/*直方图均衡化*/
	Mat imageHistogramEqualization(Mat img);
	/*指数变换增强*/
	Mat imageExponentialTransform(Mat img,double c ,double r);

	/*图像加马赛克*/
	Mat imageMasaic(Mat img, int blockSize=5);
	
	/*图像的卷积*/
	Mat imageCovolution(Mat img, int kernel_size, int** kernel);

	/*图像的傅里叶变换*/
	Mat imageFourierTransform(Mat img);

	/*图像合成*/
	Mat imageSynthesis(Mat img1,Mat img2);

	/*图像分割*/
	Mat imageSegmentation(Mat img);

	/*高级功能*/
	Mat imageDigitalIdentify(Mat img);

	/*其他功能*/
	Mat imageSketch(Mat src);



};