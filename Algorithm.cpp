#include "Algorithm.h"


/*
函数作用：滤波处理函数，根据option的不同，我们选择不同的滤波方法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：卷积核大小，默认为3
		 3、channels：图片通道数，默认为3
		 4、option：默认为AVERGE_FILTER均值滤波器,
					MEDIAN_FILTER中值滤波器,
					GAUSSIAN_FILTER高斯滤波器,
					BILATERAL_FILTER双边滤波器,
					SMALLWAVE_FILTER小波滤波器
返回值：返回经过均值滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageDenoising(Mat img, int kernel_size, int channels,int option)
{
	switch (option)
	{
	case AVERAGE_FILTER:
		return imageAverageFilter(img, kernel_size, channels);
	case MEDIAN_FILTER:
		return imageMedianFilter(img, kernel_size, channels);
	case GAUSSIAN_FILTER:
		return imageGaussianFilter(img, kernel_size, channels);
	case BILATERAL_FILTER:
		return imageBilateralFilter(img, kernel_size, channels);
	case SMALLWAVE_FILTER:
		return imageWaveletFilter(img);
	default:
		break;
	}
	return img;
}

/*
函数作用：均值滤波处理图像，让图片达到平滑效果
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：窗口大小
		 3、channels：图片通道数
返回值：返回经过均值滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageAverageFilter(Mat img, int kernel_size, int channels)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*窗口的半径*/
	int r = kernel_size / 2;
	/*根据通道数量，动态的开辟存储求和的空间*/
	double* sum = new double[channels];

	/*x，y这两层循环用于遍历图像的所有像素*/
	for (int x = r; x < img.rows - r; ++x) {
		for (int y = r; y < img.cols - r; ++y) {
			/*初始化sum*/
			for (int i = 0; i < channels; ++i)sum[i] = 0;
			/*i，j这两层循环用于遍历窗口中的元素*/
			for (int i = -r; i < r; ++i) {
				for (int j = -r; j < r; ++ j) {
					/*m这层循环是用于遍历通道数，一般彩色图片都是RGB3通道，如果传进来的是灰度图像，那么channels=1*/
					for (int m = 0; m < channels; ++m) {
						/*单通道使用单通道的读取方式，多通道使用多通道的读取方式*/
						if (channels == 1)sum[m] += img.at<uchar>(x + i, y + j);
						else if (channels > 1) sum[m] += img.at<Vec3b>(x + i, y + j)[m];
					}
				}
			}
			/*窗口中的各个通道求和已经完毕，除以窗口内总数得到均值，赋给结果Mat--imgRes*/
			for (int i = 0; i < channels; ++i) {
				if(channels>1)imgRes.at<Vec3b>(x, y)[i] = saturate_cast<uchar>(sum[i] / (kernel_size * kernel_size));
				else if (channels == 1)imgRes.at<uchar>(x, y) = saturate_cast<uchar>(sum[i] / (kernel_size * kernel_size));
			}
		}
	}

	/*释放资源*/
	delete [] sum;
	return imgRes;

}

/*
函数作用：中值滤波处理图像，去除椒盐噪声
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：窗口大小
		 3、channels：图片通道数
返回值：返回经过中值滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageMedianFilter(Mat img, int kernel_size, int channels)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*窗口的半径*/
	int r = kernel_size / 2;
	/*根据通道数量，动态的开辟存储求和的空间*/
	double* sum = new double[channels*kernel_size];

	/*x，y这两层循环用于遍历图像的所有像素*/
	for (int x = r; x < img.rows - r; ++x) {
		for (int y = r; y < img.cols - r; ++y) {
			/*初始化sum*/
			for (int i = 0; i < channels; ++i)sum[i] = 0;
			/*i，j这两层循环用于遍历窗口中的元素*/
			for (int i = -r; i < r; ++i) {
				for (int j = -r; j < r; ++j) {
					/*m这层循环是用于遍历通道数，一般彩色图片都是RGB3通道，如果传进来的是灰度图像，那么channels=1*/
					for (int m = 0; m < channels; ++m) {
						/*单通道使用单通道的读取方式，多通道使用多通道的读取方式*/
						int op_y = abs(i) + abs(j);
						if (channels == 1)sum[m*kernel_size+op_y] = img.at<uchar>(x+i, y+j);
						else if (channels > 1) sum[m * kernel_size + op_y] = img.at<Vec3b>(x+i, y+j)[m];
					}
				}
			}
			/*窗口中各个通道中的灰度值都读取完毕，我们将其进行排序，然后选择居中的进行赋值*/
			for (int i = 0; i < channels; ++i) {
				sort(sum+(i*kernel_size), sum + ((i+1)*kernel_size));
				if (channels > 1)imgRes.at<Vec3b>(x, y)[i] = saturate_cast<uchar>(sum[i * kernel_size+kernel_size/2]);
				else if (channels == 1)imgRes.at<uchar>(x, y) = saturate_cast<uchar>(sum[i * kernel_size + kernel_size / 2]);
			}
		}
	}

	/*释放资源*/
	delete[] sum;
	return imgRes;
}

/*
函数作用：高斯滤波处理图像，对图像平滑处理最好的方法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：卷积核大小
		 3、channels：图片通道数
返回值：返回经过高斯滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageGaussianFilter(Mat img, int kernel_size, int channels)
{
	return Mat();
}

/*
函数作用：进行图片的边缘提取，可用于处理物体检测和跟踪，图像分割，图像增强，噪声去噪，图像压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、order：1:一阶算子，2:二阶算子
		 3、option：一阶：1:Roberts算子，2:Sobel算子，3:Prewitt算子，4:Kirsch算子，5:Robinson算子，二阶：6:Laplacion算子，7:Canny算子
返回值：返回经过所选择的边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageEdgeDetection(Mat img,int order,int option, int denoising, int threshold)
{
	/*
	处理逻辑：根据参数order选择对应的order阶差分算子，然后根据option选择对应的边缘检测算子进行处理，返回处理后的像素矩阵
	*/
	if (order == 1) {
		switch (option)
		{
		case 1:
			return imageRoberts(img, denoising,threshold);
		case 2:
			return imageSobel(img, denoising);
		case 3:
			return imagePrewitt(img, denoising);
		case 4:
			return imageKirsch(img, denoising);
		case 5:
			return imageRobinson(img, denoising);
		default:
			break;
		}
	}
	else if (order == 2) {
		switch (option)
		{
		case 1:
			return imageLaplacian(img, denoising);
		case 2:
			return imageCanny(img, denoising);
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
Mat ImageAlgorithm::imageRoberts(Mat img, int denoising, int threshold)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	/*Roberts算子边缘检测对噪声非常敏感，所以在用算子模板进行计算前，我们对图像进行降噪处理*/
	if(denoising!=NO_FILTER)img = imageDenoising(img, 3, img.channels(), denoising);

	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*用于检测水平方向*/
	int robert_x[2][2] = { {1,0},{0,-1} };
	/*用于检测垂直方向*/
	int robert_y[2][2] = { {0,1},{-1,0} };

	for (int x = 0; x < img.rows - 1; ++x) {
		for (int y = 0; y < img.cols - 1; ++y) {
			int gx = 0, gy = 0;
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					gx += img.at<uchar>(x + i, y + j) * robert_x[i][j];
					gy += img.at<uchar>(x + i, y + j) * robert_y[i][j];
				}
			}
			int ds = sqrt(gx * gx + gy * gy);
			/*根据阈值进行二值化处理*/
			if (ds > threshold) {
				imgRes.at<uchar>(x, y) = 255;
			}
			else {
				imgRes.at<uchar>(x, y) = 0;
			}
		}
	}
	
	return imgRes;
}
/*
函数作用：Sobel算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Sobel边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageSobel(Mat img, int denoising)
{

	return Mat();
}
/*
函数作用：Prewitt算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Prewitt边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imagePrewitt(Mat img, int denoising)
{
	return Mat();
}
/*
函数作用：Kirsch算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Kirsch边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageKirsch(Mat img, int denoising)
{
	return Mat();
}

/*
函数作用：Robinson算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Robinson边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageRobinson(Mat img, int denoising)
{
	return Mat();
}
/*
函数作用：Laplacian算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Laplacian边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageLaplacian(Mat img, int denoising)
{
	return Mat();
}
/*
函数作用：Canny算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Canny边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageCanny(Mat img, int denoising)
{
	return Mat();
}
/*
函数作用：双边滤波处理图像，可用于图像降噪，边缘保留等
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：卷积核大小
		 3、channels：图片通道数
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageBilateralFilter(Mat img, int kernel_size, int channels)
{
	return Mat();
}
/*
函数作用：小波滤波处理图像，用于信号处理和图像处理中的降噪、压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageWaveletFilter(Mat img)
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
