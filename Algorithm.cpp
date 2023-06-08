#include "Algorithm.h"


/*
函数作用：进行图片的边缘提取，可用于处理物体检测和跟踪，图像分割，图像增强，噪声去噪，图像压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、order：1:一阶算子，2:二阶算子
		 3、option：一阶：1:Roberts算子，2:Sobel算子，3:Prewitt算子，4:Kirsch算子，5:Robinson算子，二阶：6:Laplacion算子，7:Canny算子
返回值：返回经过所选择的边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageEdgeDetection(Mat img,int order,int option)
{
	/*
	处理逻辑：根据参数order选择对应的order阶差分算子，然后根据option选择对应的边缘检测算子进行处理，返回处理后的像素矩阵
	*/
	if (order == 1) {
		switch (option)
		{
		case 1:
			return imageRoberts(img);
		case 2:
			return imageSobel(img);
		case 3:
			return imagePrewitt(img);
		case 4:
			return imageKirsch(img);
		case 5:
			return imageRobinson(img);
		default:
			break;
		}
	}
	else if (order == 2) {
		switch (option)
		{
		case 1:
			return imageLaplacian(img);
		case 2:
			return imageCanny(img);
		default:
			break;
		}
	}
	return Mat();
}
/*
函数作用：Roberts算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Roberts边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageRoberts(Mat img)
{
	return Mat();
}
/*
函数作用：Sobel算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Sobel边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageSobel(Mat img)
{

	return Mat();
}
/*
函数作用：Prewitt算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Prewitt边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imagePrewitt(Mat img)
{
	return Mat();
}
/*
函数作用：Kirsch算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Kirsch边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageKirsch(Mat img)
{
	return Mat();
}

/*
函数作用：Robinson算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Robinson边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageRobinson(Mat img)
{
	return Mat();
}
/*
函数作用：Laplacian算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Laplacian边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageLaplacian(Mat img)
{
	return Mat();
}
/*
函数作用：Canny算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Canny边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageCanny(Mat img)
{
	return Mat();
}
/*
函数作用：双边滤波处理图像，可用于图像降噪，边缘保留等
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageBilateralFiltering(Mat img)
{
	return Mat();
}
/*
函数作用：小波滤波处理图像，用于信号处理和图像处理中的降噪、压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageWaveletFiltering(Mat img)
{
	return Mat();
}
/*
函数作用：图像增强，根据option选择对应的算法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、option：1:对比度增强，2:亮度增强，3:直方图均衡化，4:指数变换增强
返回值：返回经过对应算法处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageEnhance(Mat img,int option)
{
	switch (option)
	{
	case 1:
		return imageContrastEnhance(img);
	case 2:
		return imageBrightness(img);
	case 3:
		return imageHistogramEqualization(img);
	case 4:
		return imageExponentialTransform(img);
	default:
		break;
	}
	return Mat();
}
/*
函数作用：对比度增强，增强图像中不同灰度级之间的差异程度，使图像更加清晰明亮。
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回对比度增强过后的像素矩阵
*/
Mat ImageAlgorithm::imageContrastEnhance(Mat img)
{
	return Mat();
}
/*
函数作用：亮度增强，提高图像的整体亮度水平，使图像更加明亮
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回亮度增强过后的像素矩阵
*/
Mat ImageAlgorithm::imageBrightness(Mat img)
{
	return Mat();
}
/*
函数作用：直方图均衡化，可以增强图像的对比度，使图像更加清晰明亮
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回直方图均衡化过后的像素矩阵
*/
Mat ImageAlgorithm::imageHistogramEqualization(Mat img)
{
	return Mat();
}
/*
函数作用：指数变换增强，对图像的灰度级进行非线性变换的方法，可以用于增强图像的局部对比度，使图像更加清晰明亮
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回指数变换增强过后的像素矩阵
*/
Mat ImageAlgorithm::imageExponentialTransform(Mat img)
{
	return Mat();
}
/*
函数作用：给图像打上马赛克
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回打上马赛克以后的图像的像素矩阵
*/
Mat ImageAlgorithm::imageMasaic(Mat img)
{
	return Mat();
}
/*
函数作用：图像卷积，可以用于图像处理中的平滑，锐化，边缘检测等任务
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageCovolution(Mat img)
{
	return Mat();
}
/*
函数作用：傅里叶变换可以将图像从空间域转化成频率域
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过傅里叶变换处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageFourierTransform(Mat img)
{
	return Mat();
}
/*
函数作用：图像融合
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回融合后图像的像素矩阵
*/
Mat ImageAlgorithm::imageSynthesis(Mat img)
{
	return Mat();
}

/*
函数作用：图像分割可以用于目标检测，图像识别，图像增强，医学影像分析等
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回分割后的图像
*/
Mat ImageAlgorithm::imageSegmentation(Mat img)
{
	return Mat();
}

/*
函数作用：识别图像中的数字
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回处理过后的像素矩阵，图像能够呈现被识别出来的数字
*/
Mat ImageAlgorithm::imageDigitalIdentify(Mat img)
{
	return Mat();
}
