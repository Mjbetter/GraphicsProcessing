#include "Algorithm.h"

/*
函数作用：图像加载函数，将传入的图片进行加载
函数参数：1、imageName：传入需要处理的图像路径
返回值：返回加载过后的像素矩阵
*/
Mat ImageAlgorithm::imageLoading_Show(String imageName)
{
	/*将图片加载后赋值到图像变量image中, 读入图像方式默认为彩色图像*/
	Mat image = imread(imageName);  
	/*检查文件是否打开（或是否为空数据），没打开时执行打印语句*/
	if (image.empty())
		cout << "Could not open or find the image" << std::endl;
	/*显示图片*/
    /*创建一个名为Image的可调节的窗口*/
	//namedWindow("Image", WINDOW_AUTOSIZE);
    /*创建一个窗口显示图像*/
	imshow("Image", image);
    /*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	waitKey(0);    

	return image;
}

/*
函数作用：图像基本变换处理函数，根据option的不同，选择不同的变换方法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		  2、dx，dy：x轴和y轴方向上的偏移量
          3、Scale_x，Scale_y：横向及纵向缩放的比例大小
		  4、angle：旋转角度
		  5、choice：镜像变换的选择数
		  5、option：表示基本变换的选择：1：图像平移，2：图像缩放，3：图像旋转，4：图像镜像
返回值：返回经过基本变换后的像素矩阵
*/
Mat ImageAlgorithm::imageNoiseAddition(Mat img, int dx, int dy, double Scale_x, double Scale_y, double angle, int choice, int option)
{
    /*处理逻辑：根据选择进行图像的基本变换，返回处理后的像素矩阵*/
    switch (option)
	{
	case 1:
		return imageTranslation(img, dx, dy); /*图像平移*/
	case 2:
		return imageResizing(img, Scale_x, Scale_y); /*图像缩放*/
	case 3:
		return imageRotating(img,/* double img_cols, double img_rows,*/angle); /*图像旋转*/
	case 4:
		return imageReflection(img, choice); /*图像镜像*/
	default:
		break;
	}
	return img;
}

/*
函数作用：图像平移函数，将图像按照输入的x方向和y方向上的偏移量进行平移，规定向右、向下时值为正数
函数参数：1、img：传入的像素矩阵
		  2、dx：在x轴方向上的位移，dx>0时，向右平移X个单位
		  3、dy：在y轴方向上的位移，dy>0时，向下平移Y个单位
返回值：返回平移过后的像素矩阵
*/
Mat ImageAlgorithm::imageTranslation(Mat img, int dx, int dy)
{
	/*获取图像的形状信息*/
	Size imageSize = img.size();
	/*创建一个2X3的浮点型仿射变换矩阵以及一个存储平移后的像素矩阵*/
	Mat translationMatrix = (Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
	Mat translatedImage;
	warpAffine(img, translatedImage,translationMatrix, img.size());
	//translatedImage = warpAffine(img, translationMatrix, img.size());
	/*创建一个名为Image的可调节的窗口*/
	//namedWindow("translatedImage", WINDOW_AUTOSIZE);
	/*创建一个窗口显示图像*/
	imshow("translatedImage", translatedImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	waitKey(0);

	return translatedImage;
}

/*
函数作用：图像缩放函数，将图像按照输入的比例进行缩放操作
函数参数：1、img：传入的像素矩阵
		  2、Scale_x：横向缩放的比例大小
		  3、Scale_y：纵向缩放的比例大小
返回值：返回缩放过后的像素矩阵
*/
Mat ImageAlgorithm::imageResizing(Mat img, double Scale_x, double Scale_y)
{

	Mat resizingImage;
	/*按横向和纵向比例缩放图像*/
	resize(img, resizingImage, Size(), Scale_x, Scale_y);
	/*创建一个窗口显示图像*/
	imshow("resizingImage", resizingImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	waitKey(0);

	return resizingImage;
}

/*
函数作用：图像旋转函数，将图像按照输入的旋转角度
函数参数：1、img：传入的像素矩阵
          2、img_x：旋转点的x坐标
		  3、img_y：旋转点的y坐标
		  4、angle：输入的旋转角度（以逆时针方向为正）
返回值：返回旋转过后的像素矩阵
*/
Mat ImageAlgorithm::imageRotating(Mat img/*, double img_cols, double img_rows*/, double angle)
{
	/*创建用于存储旋转后的像素矩阵*/
	Mat rotatingImage;
	/*图像中心点*/
	//Point2f center(img_cols / 2.0, img_rows / 2.0);
	/*创建旋转矩阵，参数分别为旋转点坐标，旋转角度，缩放因子（设置为1.0，表示不进行缩放）*/
	Mat rotatingMatrix = getRotationMatrix2D(Point2f(img.cols / 2.0, img.rows / 2.0), angle, 1.0);
	/*将旋转矩阵应用于图像实现图像旋转*/
	warpAffine(img, rotatingImage, rotatingMatrix, img.size());
	/*创建一个窗口显示图像*/
	imshow("rotatingImage", rotatingImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	waitKey(0);

	return rotatingImage;
}

/*
函数作用：镜像变换函数，通过0 1实现图像水平和垂直镜像
函数参数：1、img：传入的像素矩阵
		  2、choice：数字0或1用于判断实现哪种镜像
返回值：返回镜像变换过后的像素矩阵
*/
Mat ImageAlgorithm::imageReflection(Mat img, int choice)
{
	/*创建用于存储镜像变换后的像素矩阵*/
	Mat reflectionImage;
    /*判断choice值是否合法*/
	if (choice != 1 && choice != 0) {
		return reflectionImage;
	}
	/*若choice值合法*/
	else {
		/*垂直镜像翻转图像*/
		if (choice == 0)
			flip(img, reflectionImage, 0);
		/*水平镜像翻转图像*/
		else if (choice == 1)
			flip(img, reflectionImage, 1);
	}
	/*创建一个窗口显示图像*/
	imshow("rotatingImage", reflectionImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	waitKey(0);

	return reflectionImage;
}

/*
函数作用：图像变灰度处理函数，根据option的不同，选择不同的变换方法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		  2、dx，dy：x轴和y轴方向上的偏移量
		  3、Scale_x，Scale_y：横向及纵向缩放的比例大小
		  4、angle：旋转角度
		  5、choice：镜像变换的选择数
		  5、option：表示基本变换的选择：1：图像平移，2：图像缩放，3：图像旋转，4：图像镜像
返回值：返回经过基本变换后的像素矩阵
*/
//Mat ImageAlgorithm::imageGrayscale(Mat img, int dx, int dy, double Scale_x, double Scale_y, double angle, int choice, int option)
//{
//	/*处理逻辑：根据选择进行图像的基本变换，返回处理后的像素矩阵*/
//	switch (option)
//	{
//	case 1:
//		return imageTranslation(img, dx, dy); /*图像平移*/
//	case 2:
//		return imageResizing(img, Scale_x, Scale_y); /*图像缩放*/
//	case 3:
//		return imageRotating(img,/* double img_cols, double img_rows,*/angle); /*图像旋转*/
//	case 4:
//		return imageReflection(img, choice); /*图像镜像*/
//	default:
//		break;
//	}
//	return img;
//}


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
		5、level：小波去噪的分解层数，默认为3
返回值：返回经过均值滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageDenoising(Mat img, int kernel_size, int channels,int option,int level)
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
		return imageWaveletFilter(img,level);
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
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*窗口半径*/
	int r = kernel_size / 2;
	/*根据窗口大小和通道数量动态分配窗口内存空间*/
	double** kernel = new double* [kernel_size];
	double sum = 0;
	double sigma = kernel_size / 6.0;
	for (int i = 0; i < kernel_size; ++i) {
		kernel[i] = new double[kernel_size];
	}

	/*根据高斯函数，生成高斯卷积模板*/
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			kernel[i][j] = exp(-(pow(i - r, 2) + pow(j - r, 2)) / (2 * pow(sigma, 2)));
			sum += kernel[i][j];
		}
	}
	/*归一化处理，使模板总和=1，控制卷积强度，保留图像细节和特征，避免出现图像的过渡增强或者失真*/
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			kernel[i][j] /= sum;
		}
	}

	double* kernel_sum = new double[channels];
	for (int i = 0; i < channels; ++i) kernel_sum[i] = 0;
	/*x,y这两层循环遍历图片所有像素*/
	for (int x = 0; x < img.rows-kernel_size; ++x) {
		for (int y = 0; y < img.cols - kernel_size; ++y) {
			/*i,j这两层循环遍历窗口进行求和处理*/
			for (int i = 0; i < kernel_size; ++i) {
				for (int j = 0; j < kernel_size; ++j) {
					/*m这层循环主要考虑通道数量*/
					for (int m = 0; m < channels; ++m) {
						if (channels == 1) {
							kernel_sum[m] += img.at<uchar>(x + i, y + j) * kernel[kernel_size - i -1][kernel_size - j -1];
						}
						else if (channels > 1) {
							kernel_sum[m] += img.at<Vec3b>(x + i, y + j)[m] * kernel[kernel_size - i - 1][kernel_size - j - 1];
						}
					}
				}
			}
			/*卷积完成以后，将该和赋给对应点位*/
			for (int i = 0; i < channels; ++i) {
				if (channels == 1) imgRes.at<uchar>(x, y) = kernel_sum[i];
				else if (channels > 1)imgRes.at<Vec3b>(x, y)[i] = kernel_sum[i];
				kernel_sum[i] = 0;
			}
		}
	}

	/*释放资源*/
	for (int i = 0; i < kernel_size; ++i) {
		delete[] kernel[i];
	}
	delete[] kernel;
	delete[] kernel_sum;
	return imgRes;
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
	Mat imgRes;
	imgRes.create(img.size(), img.type());

	/*窗口半径*/
	int r = kernel_size / 2;
	double sigma = kernel_size / 6.0;

	/*通道灰度值的求和*/
	double* sum = new double[channels];
	/*通道权重的求和*/
	double* weight_sum = new double[channels];
	for (int i = 0; i < channels; ++i) {
		sum[i] = 0;
		weight_sum[i] = 0;
	}
	/*x,y循环遍历图像的所有像素*/
	for (int x = 0; x < img.rows-kernel_size; ++x) {
		for (int y = 0; y < img.cols - kernel_size; ++y) {
			/*i,j循环遍历窗口*/
			for (int i = 0; i < kernel_size; ++i) {
				for (int j = 0; j < kernel_size; ++j) {
					/*m循环遍历通道*/
					for (int m = 0; m < channels; ++m) {
						if (channels == 1) {
							/*空间权重*/
							double space_weight = exp(-(i-r)*(i-r)+(j-r)*(j-r))/(2*sigma*sigma);
							/*灰度值权重*/
							double color_weight = exp(-(img.at<uchar>(x+i,y+j)-img.at<uchar>(x,y))* (img.at<uchar>(x + i, y + j) - img.at<uchar>(x, y)));
							/*双边滤波权重*/
							double weight = color_weight * space_weight;
							/*对权重求和的操作，便于进行归一化处理*/
							sum[m] += img.at<uchar>(x+i,y+j) * weight;
							/*对结果进行求和*/
							weight_sum[m] += weight;
						}
						else if (channels > 1) {
							/*空间权重*/
							double space_weight = exp(-(i - r) * (i - r) + (j - r) * (j - r)) / (2 * sigma * sigma);
							/*灰度值权重*/
							double color_weight = exp(-(img.at<Vec3b>(x + i, y + j)[m] - img.at<Vec3b>(x, y)[m])* (img.at<Vec3b>(x + i, y + j)[m] - img.at<Vec3b>(x, y)[m]));
							/*双边滤波权重*/
							double weight = color_weight * space_weight;
							/*对权重求和的操作，便于进行归一化处理*/
							sum[m] += img.at<Vec3b>(x + i, y + j)[m] * weight;
							/*对结果进行求和*/
							weight_sum[m] += weight;
						}
					}
				}
			}
			/*进行归一化处理，并将结果赋给输出矩阵*/
			for (int i = 0; i < channels; ++i) {
				if (channels == 1) {
					imgRes.at<uchar>(x, y) = sum[i] / weight_sum[i];
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(x, y)[i] = sum[i] / weight_sum[i];
				}
				sum[i] = 0;
				weight_sum[i] = 0;
			}
		}
	}
	return imgRes;
}

/*
函数作用：小波变换，将图像拆分成多个不同频带的子带，从而实现去噪的目的。
参数：1、data：为要被拆分的频带，之所为为二维，是因为彩色图像是多通道
	  2、lowPass：存储低频信号
	  3、highPass：存储高频信号
没有返回值，因为对data的改变会直接作用到data上，传入的是地址
*/
void ImageAlgorithm::waveLetTransform(double** data, double ** lowPass,double **highPass,int rows,int cols)
{
	/*分别通过平滑处理和差分处理得到aC和aD*/
	for (int i = 0; i < cols/2; ++i) {
		for (int j = 0; j < rows; ++j) {
			/*通过移动平均进行平滑处理*/
			lowPass[j][i] = (data[j][2 * i] + data[j][2 * i + 1]) / sqrt(2);
			/*通过一阶差分进行平滑处理*/
			highPass[j][i] = (data[j][2 * i] - data[j][2 * i + 1]) / sqrt(2);
		}
	}
}

/*
函数作用：计算噪声方差
函数参数：1、data：高频带信号
		2、threashold:阈值数组，对应不同的通道
*/
void ImageAlgorithm::getThreashold(double** data,double *threashold,int rows, int cols)
{
	double** absData = new double* [rows];
	for (int i = 0; i < rows; ++i) {
		absData[i] = new double[cols];
	}

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			absData[i][j] = abs(data[i][j]);
		}
	}

	/*对序列进行排序*/
	for (int i = 0; i < rows; ++i) {
		sort(absData[i], absData[i] + cols);
		/*对序列去取中值*/
		double median = absData[i][cols / 2];
		double sigma = median / 0.6745;
		threashold[i] = sigma * sigma;
	}

	for (int i = 0; i < rows; ++i) {
		delete[] absData[i];
	}
	delete[] absData;
}

/*
函数作用：进行阈值处理
参数：1、data：需要进行处理的信号
	  2、threadshold：阈值
无返回值
*/
void ImageAlgorithm::doThreashold(double** data, double *threadshold,int rows,int cols)
{
	for (int i = 0; i < cols; ++i) {
		for (int j = 0; j < rows; ++j) {
			if (abs(data[j][i]) < threadshold[j]) {
				data[j][i] = 0;
			}
		}
	}
}

/*
函数作用：小波反变换，将多个不同频带的信号合在一起，重构图像
参数：1、data：为要被拆分的频带，之所为为二维，是因为彩色图像是多通道
	  2、lowPass：存储低频信号
	  3、highPass：存储高频信号
没有返回值，因为对data的改变会直接作用到data上，传入的是地址
*/
void ImageAlgorithm::inverseWaveLetTransform(double** data, double** lowPass, double** highPass, int rows, int cols)
{
	for (int i = 0; i < cols/2; ++i) {
		for (int j = 0; j < rows; ++j) {
			data[j][i * 2] = (lowPass[j][i] + highPass[j][i]) / sqrt(2);
			data[j][i * 2+1] = (lowPass[j][i] - highPass[j][i]) / sqrt(2);
		}
	}
}

/*
函数作用：小波滤波处理图像，用于信号处理和图像处理中的降噪、压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、level：小波去噪的层数
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageWaveletFilter(Mat img, int level)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*得到图像的通道数*/
	int channels = img.channels();
	/*图片像素总数,只考虑单通道*/
	int data_num = img.cols * img.rows;
	/*将图像转化成一维数组,根据通道数量进行不同的处理*/
	double** data = new double* [channels];
	for (int i = 0; i < channels; ++i) {
		data[i] = new double[data_num];
	}
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) { 
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					data[m][i * img.cols + j] = img.at<uchar>(i, j);
				}
				else if (channels > 1) {
					data[m][i * img.cols + j] = img.at<Vec3b>(i, j)[m];
				}
			}
		}
	}
	double *threashold = new double[channels];

	/*给高频和低频的存储开辟对应的内存空间*/
	double** row = new double* [channels];
	double** lowPass = new double* [channels];
	double** highPass = new double* [channels];
	for (int i = 0; i < channels; ++i) {
		lowPass[i] = new double[data_num / 2];
		highPass[i] = new double[data_num - data_num / 2];
		row[i] = new double[data_num];

	}

	int size_i = img.rows;
	int size_j = img.cols;

	/*i循环是针对level进行小波变换的次数*/
	for (int l = 0; l < level; ++l) {
		/*进行小波变换*/
		for (int i = 0; i < size_i; ++i) {
			for (int j = 0; j < size_j; ++j) {
				for (int m = 0; m < channels; ++m) {
					row[m][i * size_i + j] = data[m][i * size_i + j];
				}
			}
		}
		/*进行小波变换*/
		waveLetTransform(row, lowPass, highPass, channels, size_j * size_i);
		/*获得阈值*/
		getThreashold(highPass,threashold, channels, size_j * size_i/2);
		for (int i = 0; i < channels; ++i) {
			threashold[i] = threashold[i] * sqrt(2 * log(size_i * size_j));
		}
		/*进行阈值处理*/
		doThreashold(highPass, threashold, channels, size_j * size_i/2);
		/*进行小波反变换*/
		inverseWaveLetTransform(row, lowPass, highPass, channels, size_j*size_i);
		/*将这次处理后的值给原矩阵*/
		for (int i = 0; i < size_i; ++i) {
			for (int j = 0; j < size_j; ++j) {
				for (int m = 0; m < channels; ++m) {
					data[m][i * size_j + j] = row[m][i * size_j + j];
				}
			}
		}
		/*每多一层，那么cA就会减少一半*/
		size_i /= 2;
		size_j /= 2;
	}

	/*将一维数组重新转化成二维数组*/
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(i, j) = data[m][i * img.cols + j];
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(i, j)[m] = data[m][i * img.cols + j];
				}
			}
		}
	}

	/*资源释放*/
	for (int i = 0; i < channels; ++i) {
		delete[] lowPass[i];
		delete[] highPass[i];
		delete[] data[i];
	}
	delete[] lowPass;
	delete[] highPass;
	delete[] row;
	delete[] data;
	delete[] threashold;

	return imgRes;
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
Mat ImageAlgorithm::imageSobel(Mat img, int denoising,int kernel_size)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	/*图像的去噪平滑处理*/
	if (denoising != NO_FILTER)img = imageDenoising(img, 3, img.channels(), denoising);

	Mat imgRes;
	imgRes.create(img.size(), img.type());
 	double kernel_three_x[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
	double kernel_three_y[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	double kernel_five_x[5][5] = { {-1,-2,0,2,1},{-4,-8,0,8,4},{-6,-12,0,12,6},{-4,-8,0,8,4},{-1,-2,0,2,1} };
	double kernel_five_y[5][5] = { {-1,-4,-6,-4,-1},{-2,-8,-12,-8,-2},{0,0,0,0,0},{2,8,12,8,2},{1,4,6,4,1} };
	int threashold = 0;
	if (kernel_size == 3) {
		kernel_size = 3;
		threashold = 80;
	}
	else {
		kernel_size = 5;
		threashold = 2000;
	}
	int r = kernel_size / 2;

	/*x,y这两层循环遍历图像的所有像素*/
	for (int x = r; x < img.rows - r; ++x) {
		for (int y = r; y < img.cols - r; ++y) {
			/*i,j这两层循环对模板进行一个遍历*/
			double g_x = 0.0, g_y = 0.0;
			for (int i = -r; i <= r; ++i) {
				for (int j = -r; j <= r; ++j) {
					if (kernel_size == 3) {
						g_x += img.at<uchar>(x+i,y+j) * kernel_three_x[i + 1][j + 1];
						g_y += img.at<uchar>(x + i, y + j) * kernel_three_y[i + 1][j + 1];
					}
					else if (kernel_size == 5) {
						g_x += img.at<uchar>(x + i, y + j) * kernel_five_x[i + 1][j + 1];
						g_y += img.at<uchar>(x + i, y + j) * kernel_five_y[i + 1][j + 1];
					}
				}
			}
			int ds = sqrt(g_x * g_x + g_y * g_y);
			/*根据阈值进行二值化处理*/
			if (ds > threashold) {
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
函数作用：Prewitt算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Prewitt边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imagePrewitt(Mat img, int denoising)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	/*图像平滑去噪*/
	if (denoising != NO_FILTER)img = imageDenoising(img, 3, img.channels(), denoising);
	Mat imgRes;
	imgRes.create(img.size(), img.type());

	/*Prewitt算子模板*/
	int kernel_x[3][3] = { {-1,0,1},{-1,0,1},{-1,0,1} };
	int kernel_y[3][3] = { {-1,-1,-1},{0,0,0},{1,1,1} };

	/*x,y这两层循环用于遍历图像的所有像素*/
	for (int x = 1; x < img.rows-1; ++x) {
		for (int y = 1; y < img.cols-1; ++y) {
			/*i,j这两层循环用于遍历Prewitt算子模板*/
			double G_x=0.0, G_y = 0.0;
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					G_x += img.at<uchar>(x + i, y + j)*kernel_x[i + 1][j + 1];
					G_y += img.at<uchar>(x + i, y + j) * kernel_y[i + 1][j + 1];
				}
			}
			/*对水平方向和垂直方向的灰度值进行处理，Gx+Gy=该点灰度值*/
			double grad = abs(G_x) + abs(G_y);
			grad = grad > 255 ? 255 : grad;
			grad = grad < 0 ? 0 : grad;
			imgRes.at<uchar>(x, y) = grad;
		}
	}
	return imgRes;
}

/*
函数作用：Kirsch算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Kirsch边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageKirsch(Mat img, int denoising)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	/*图像平滑去噪*/
	if (denoising != NO_FILTER)img = imageDenoising(img, 3, img.channels(), denoising);
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*八个方向的卷积模板*/
	int kirsch[8][3][3] = {
		{{-3,-3,5},{-3,0,5},{-3,-3,5}},
		{{-3,5,5},{-3,0,5},{-3,-3,-3}},
		{{5,5,5},{-3,0,-3},{-3,-3,-3}},
		{{5,5,-3},{5,0,-3},{-3,-3,-3}},
		{{5,-3,-3},{5,0,-3},{5,-3,-3}},
		{{-3,-3,-3},{5,0,-3},{5,5,-3}},
		{{-3,-3,-3},{-3,0,-3},{5,5,5}},
		{{-3,-3,-3},{-3,0,5},{-3,5,5}}
	};
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			double max_val = 0;
			double direction = 0;
			for (int i = 0; i < 8; ++i) {
				double val = 0;
				for (int j = -1; j <= 1; ++j) {
					for (int k = -1; k <= 1; ++k) {
						val += img.at<uchar>(x + j, y + k) * kirsch[i][j + 1][k + 1];
					}
				}
				if (abs(val) > max_val) {
					max_val = abs(val);
				}
			}
			if (max_val > 255)max_val = 255;
			else if (max_val < 0)max_val = 0;
			imgRes.at<uchar>(x, y) = max_val;
		}
	}
	return imgRes;
}

/*
函数作用：Robinson算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Robinson边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageRobinson(Mat img, int denoising)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	/*图像平滑去噪*/
	if (denoising != NO_FILTER)img = imageDenoising(img, 3, img.channels(), denoising);
	Mat imgRes;
	imgRes.create(img.size(), img.type());

	int threashold = 100;

	/*八个方向的卷积模板*/
	int robinson[8][3][3] =
	{
		{ { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } },
		{ { 0, 1, 2 }, { -1, 0, 1 }, { -2, -1, 0 } },
		{ { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } },
		{ { -2, -1, 0 }, { -1, 0, 1 }, { 0, 1, 2 } },
		{ { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } },
		{ { 0, -1, -2 }, { 1, 0, -1 }, { 2, 1, 0 } },
		{ { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } },
		{ { 2, 1, 0 }, { 1, 0, -1 }, { 0, -1, -2 } }
	};
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			double max_val = 0;
			double direction = 0;
			for (int i = 0; i < 8; ++i) {
				double val = 0;
				for (int j = -1; j <= 1; ++j) {
					for (int k = -1; k <= 1; ++k) {
						val += img.at<uchar>(x + j, y + k) * robinson[i][j + 1][k + 1];
					}
				}
				if (abs(val) > max_val) {
					max_val = abs(val);
				}
			}
			/*阈值处理*/
			if (max_val > threashold) {
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
函数作用：Laplacian算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Laplacian边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageLaplacian(Mat img, int denoising)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	/*图像平滑去噪*/
	if (denoising != NO_FILTER)img = imageDenoising(img, 3, img.channels(), denoising);
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	// 定义拉普拉斯算子
	int laplacian_kernel[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };

	for (int x = 1; x < img.rows-1; ++x) {
		for (int y = 1; y < img.cols-1; ++y) {
			double sum = 0;
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					sum += laplacian_kernel[i + 1][j + 1] * img.at<uchar>(x + i, y + j);
				}
			}
			if (sum > 255)sum = 255;
			else if (sum < 0)sum = 0;
			imgRes.at<uchar>(x, y) = sum;
		}
	}
	return imgRes;
}

/*
函数作用：Canny算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Canny边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageCanny(Mat img)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*Gaussian滤波进行图像平滑去噪*/
	img = imageGaussianFilter(img, 3, 1);
	double** mag = new double* [img.rows];
	double** dir = new double* [img.rows];
	for (int i = 0; i < img.rows; ++i) {
		mag[i] = new double[img.cols];
		dir[i] = new double[img.cols];
	}
	/*通过Sobel边缘检测计算梯度幅值和梯度方向*/
	double kernel_three_x[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
	double kernel_three_y[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			double g_x = 0, g_y = 0;
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					g_x += img.at<uchar>(x + i, y + j) * kernel_three_x[i + 1][j + 1];
					g_y += img.at<uchar>(x + i, y + j) * kernel_three_y[i + 1][j + 1];
				}
			}
			mag[x][y] = sqrt(g_x * g_x + g_y * g_y);
			/*乘180除以PAI是为了以度数来表示方向，如果不做这个处理，那么就是弧度来表示方向*/
			dir[x][y] = atan2(g_y, g_x);
		}
	}
	/*用于记录抑制后的边缘*/
	unsigned char* nmsImgData = new  unsigned char[img.rows*img.cols];
	/*非极大值抑制*/
	for (int i = 1; i < img.rows - 1; ++i) {
		for (int j = 1; j <= img.cols - 1; ++j) {
			/*将i,j这个点的像素值同梯度的其他像素值进行比较，如果不是最大值，那么进行抑制*/
			double val = mag[i][j];
			double direction = dir[i][j];
			double left=0, right=0;
			/*水平方向*/
			if (direction < -PAI / 8 && direction >= 7 * PAI / 8 ) {
				left = mag[i][j - 1];
				right = mag[i][j + 1];
			}
			/*反水平方向*/
			else if (direction >= -PAI / 8 && direction < PAI / 8) {
				left = mag[i][j - 1];
				right = mag[i][j + 1];
			}
			/*垂直方向*/
			else if (direction >= PAI / 8 && direction < 3 * PAI / 8) {
				left = mag[i + 1][j - 1];
				right = mag[i - 1][j + 1];
			}
			/*反垂直方向*/
			else{
				left = mag[i + 1][j - 1];
				right = mag[i - 1][j + 1];
			}
			if (val < left || val < right) {
				nmsImgData[i*img.cols+j] = 0;
			}
			else {
				nmsImgData[i * img.cols + j] = (unsigned char)val;
			}
 		}
	}
	/*进行双阈值处理,找出强边缘，弱边缘和非边缘*/
	double lowThreshold = 50.0;
	double highThreshold = 100.0;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (nmsImgData[i * img.cols + j] >= highThreshold) {
				imgRes.at<uchar>(i, j) = 255;
			}
			else if (nmsImgData[i * img.cols + j] >= lowThreshold) {
				imgRes.at<uchar>(i, j) = 127;
			}
			else {
				imgRes.at<uchar>(i, j) = 0;
			}
		}
	}
	/*进行双阈值中间像素滤除或者连接*/
	for (int i = 1; i < img.rows - 1; ++i) {
		for (int j = 1; j < img.cols - 1; ++j) {
			if (imgRes.at<uchar>(i, j) == 127) {
				if (imgRes.at<uchar>(i - 1, j - 1) == 255 ||
					imgRes.at<uchar>(i, j - 1) == 255 ||
					imgRes.at<uchar>(i - 1, j) == 255 ||
					imgRes.at<uchar>(i, j + 1) == 255 ||
					imgRes.at<uchar>(i + 1, j - 1) == 255 ||
					imgRes.at<uchar>(i + 1, j) == 255 ||
					imgRes.at<uchar>(i + 1, j + 1) == 255 || 
					imgRes.at<uchar>(i - 1, j + 1) == 255) {
					imgRes.at<uchar>(i, j) = 255;
				}
				else {
					imgRes.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return imgRes;
}

/*
函数作用：图像增强，根据option选择对应的算法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、option：1:对比度增强，2:亮度增强，3:直方图均衡化，4:指数变换增强
		 3、L：对比度增强，亮度增强，指数增强的参数。
返回值：返回经过对应算法处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageEnhance(Mat img,int option,double L)
{
	switch (option)
	{
	case 1:
		return imageContrastEnhance(img, L);
	case 2:
		return imageBrightness(img,L);
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
		 2、L：新的灰度值范围
返回值：返回对比度增强过后的像素矩阵
*/
Mat ImageAlgorithm::imageContrastEnhance(Mat img, double L)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	int channels = img.channels();
	/*开辟对应通道数量的空间*/
	double* p_min = new double[channels];
	double* p_max = new double[channels];
	for (int i = 0; i < channels; ++i) {
		p_min[i] = 0;
		p_max[i] = 0;
	}
	/*计算像素的最大值和最小值*/
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				double tmp=0;
				if (channels == 1)tmp = img.at<uchar>(i, j);
				else if (channels > 1) tmp = img.at<Vec3b>(i, j)[m];
				if (p_min[m] > tmp) p_min[m] = tmp;
				else if (p_max[m] < tmp) p_max[m] = tmp;
			}
		}
	}
	/*旧的像素值范围*/
	double* old_range = new double[channels];
	for (int i = 0; i < channels; ++i) {
		old_range[i] = p_max[i] - p_min[i];
	}
	/*对像素进行线性变换*/
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				double value;
				if (channels == 1) {
					value = (img.at<uchar>(i, j) - p_min[m]) / old_range[m] * L;
				}
				else if (channels > 1) {
					value = (img.at<Vec3b>(i, j)[m] - p_min[m]) / old_range[m] * L;
				}
				if (value < 0)value = 0;
				else if (value > 255) value = 255;
				if (channels == 1) {
					imgRes.at<uchar>(i, j) = value;
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(i, j)[m] = value;
				}
			}
		}
	}
	delete []p_min;
	delete []p_max;
	delete []old_range;
	return imgRes;
}
/*
函数作用：亮度增强，提高图像的整体亮度水平，使图像更加明亮
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、L：需要增强的亮度
返回值：返回亮度增强过后的像素矩阵
*/
Mat ImageAlgorithm::imageBrightness(Mat img, double L)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	int channels = img.channels();
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				double tmp = 0;
				if (channels == 1) {
					double tmp = img.at<uchar>(i, j) + L;
					tmp = tmp > 255 ? 255 : (tmp < 0 ? 0 : tmp);
					imgRes.at<uchar>(i, j) = tmp;
				}
				else if (channels > 1) {
					double tmp = img.at<Vec3b>(i, j)[m] + L;
					tmp = tmp > 255 ? 255 : (tmp < 0 ? 0 : tmp);
					imgRes.at<Vec3b>(i, j)[m] = tmp;
				}

			}
		}
	}
	return imgRes;
}
/*
函数作用：统计直方图。
函数参数：1、img：需要进行统计的图像
		 2、hist：统计结果的存储数组
*/
void ImageAlgorithm::imgageStatisticalHistogram(Mat img, int** hist)
{
	int channels = img.channels();
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				int pos = 0;
				if (channels == 1)pos = img.at<uchar>(i, j);
				else pos = img.at<Vec3b>(i, j)[m];
				hist[m][pos]++;
			}
		}
	}
}
/*
函数作用：直方图均衡化，可以增强图像的对比度，使图像更加清晰明亮
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回直方图均衡化过后的像素矩阵
*/
Mat ImageAlgorithm::imageHistogramEqualization(Mat img)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	int channels = img.channels();
	int max_size = 256;
	int** hist = new int* [channels];
	int** cdf = new int* [channels];
	int** map = new int* [channels];
	for (int i = 0; i < channels; ++i) {
		hist[i] = new int[max_size];
		cdf[i] = new int[max_size];
		map[i] = new int[max_size];
		for (int j = 0; j < max_size; ++j) {
			hist[i][j] = 0;
			cdf[i][j] = 0;
			map[i][j] = 0;
		}
	}

	/*统计直方图*/
	imgageStatisticalHistogram(img,hist);

	/*计算累计概率分布函数CDF*/
	for (int i = 1; i < max_size; ++i) {
		for (int m = 0; m < channels; ++m) {
			cdf[m][i] = cdf[m][i - 1] + hist[m][i];
		}
	}

	/*计算缩放因子*/
	double scale = (max_size-1.0) / (img.rows * img.cols);

	/*计算映射表*/
	for (int i = 0; i < max_size; ++i) {
		for (int m = 0; m < channels; ++m) {
			/*为了避免浮点数的产生，使用cvRound进行取整*/
			map[m][i] = cvRound(cdf[m][i] * scale);
		}
	}

	/*映射像素值*/
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				int pos = 0;
				if (channels == 1) {
					pos = img.at<uchar>(i, j);
					imgRes.at<uchar>(i, j) = map[m][pos];
				}
				else {
					pos = img.at<Vec3b>(i, j)[m];
					imgRes.at<Vec3b>(i, j)[m] = map[m][pos];
				}
			}
		}
	}

	for (int i = 0; i < channels; ++i) {
		delete[] hist[i];
		delete[] cdf[i];
		delete[] map[i];
	}
	delete[] hist;
	delete[] map;
	delete[] cdf;

	return imgRes;
}
/*
函数作用：指数变换增强，对图像的灰度级进行非线性变换的方法，可以用于增强图像的局部对比度，使图像更加清晰明亮
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回指数变换增强过后的像素矩阵
*/
Mat ImageAlgorithm::imageExponentialTransform(Mat img)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	int channels = img.channels();

	double c = 1.1;
	double r = 1.05;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					double tmp = c * pow(img.at<uchar>(i, j), r);
					tmp = tmp > 255 ? 255 : (tmp < 0 ? 0 : tmp);
					imgRes.at<uchar>(i, j) = tmp;
				}
				else if (channels > 1) {
					double tmp = c * pow(img.at<Vec3b>(i, j)[m], r);
					tmp = tmp > 255 ? 255 : (tmp < 0 ? 0 : tmp);
					imgRes.at<Vec3b>(i, j)[m] = tmp;
				}
			}
		}
	}
	return imgRes;
}
/*
函数作用：给图像打上马赛克
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、blockSize：马赛克的大小,矩阵边长
返回值：返回打上马赛克以后的图像的像素矩阵
*/
Mat ImageAlgorithm::imageMasaic(Mat img,int blockSize)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	int channels = img.channels();

	double* sum = new double[channels];

	/*遍历图像像素*/
	for (int i = 0; i < img.rows; i+=blockSize) {
		for (int j = 0; j < img.cols; j+=blockSize) {
			/*遍历马赛克大小*/
			for (int x = i; x < i + blockSize; ++x) {
				if (x >= img.rows) break;
				for (int y = j; y < j + blockSize; ++y) {
					if (y >= img.cols) break;
					/*遍历通道大小*/
					for (int m = 0; m < channels; ++m) {
						if (channels == 1)sum[m] += img.at<uchar>(x, y);
						else sum[m] += img.at<Vec3b>(x, y)[m];
					}
				}
			}
			/*求得均值*/
			for (int m = 0; m < channels; ++m) {
				sum[m] /= blockSize * blockSize;
			}
			/*将马赛克方块内的灰度值存到结果矩阵*/
			for (int x = 0; x <  blockSize; ++x) {
				if ((i + x) >= img.rows) break;
				for (int y = 0; y <  blockSize; ++y) {
					if ((j + y) >= img.cols) break;
					for (int m = 0; m < channels; ++m) {
						if (channels == 1)imgRes.at<uchar>(x + i, y + j) = sum[m];
						else imgRes.at<Vec3b>(x + i, y + j)[m] = sum[m];
					}
				}
			}
			for (int i = 0; i < channels; ++i) {
				sum[i] = 0;
			}
		}
	}

	delete[] sum;
	return imgRes;
}
/*
函数作用：图像卷积，可以用于图像处理中的平滑，锐化，边缘检测等任务
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageCovolution(Mat img, int kernel_size, int **kernel)
{
	Mat imgRes;
	int crow = img.rows - kernel_size + 1;
	int ccol = img.cols - kernel_size + 1;
	imgRes.create(crow,ccol, img.type());
	int channels = img.channels();
	double* sum = new double[channels];
	for (int x = 0; x < crow; ++x) {
		for (int y = 0; y < ccol; ++y) {
			for (int i = 0; i < kernel_size; ++i) {
				for (int j = 0; j < kernel_size; ++j) {
					for (int m = 0; m < channels; ++m) {
						if (channels == 1)sum[m] += img.at<uchar>(x + i, y + j) * kernel[i][j];
						else sum[m] += img.at<Vec3b>(x + i, y + j)[m] * kernel[i][j];
					}
				}
			}
			for (int m = 0; m < channels; ++m) {
				sum[m] = sum[m] > 255 ? 255 : (sum[m] < 0 ? 0 : sum[m]);
				if (channels == 1)imgRes.at<uchar>(x, y) = sum[m];
				else imgRes.at<Vec3b>(x, y)[m] = sum[m];
				sum[m] = 0;
			}
		}
	}
	delete[] sum;
	return imgRes;
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
