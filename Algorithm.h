#pragma once
#include <opencv2\opencv.hpp>//加载OpenCV 4.0头文件
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

class ImageAlgorithm
{
public:
	/*基础功能*/



	/*中级功能*/

	/*图像的边缘提取*/
	Mat imageEdgeDetection(Mat img,int order,int option);
	/*一阶边缘检测算子Roberts*/
	Mat imageRoberts(Mat img);
	/*一阶边缘检测算子Sobel*/
	Mat imageSobel(Mat img);
	/*一阶边缘检测算子Prewitt*/
	Mat imagePrewitt(Mat img);
	/*一阶边缘检测算子Kirsch*/
	Mat imageKirsch(Mat img);
	/*一阶边缘检测算子Robinson*/
	Mat imageRobinson(Mat img);
	/*二阶边缘检测算子Laplacian算子*/
	Mat imageLaplacian(Mat img);
	/*二阶边缘检测算子Canny*/
	Mat imageCanny(Mat img);

	/*图像去噪进阶*/
	/*双边滤波*/
	Mat imageBilateralFiltering(Mat img);
	/*小波滤波*/
	Mat imageWaveletFiltering(Mat img);

	/*图像的增强*/
	Mat imageEnhance(Mat img,int option);
	/*对比度增强*/
	Mat imageContrastEnhance(Mat img);
	/*亮度增强*/
	Mat imageBrightness(Mat img);
	/*直方图均衡化*/
	Mat imageHistogramEqualization(Mat img);
	/*指数变换增强*/
	Mat imageExponentialTransform(Mat img);

	/*图像加马赛克*/
	Mat imageMasaic(Mat img);
	
	/*图像的卷积*/
	Mat imageCovolution(Mat img);

	/*图像的傅里叶变换*/
	Mat imageFourierTransform(Mat img);

	/*图像合成*/
	Mat imageSynthesis(Mat img);

	/*图像分割*/
	Mat imageSegmentation(Mat img);

	/*高级功能*/
	Mat imageDigitalIdentify(Mat img);


};

