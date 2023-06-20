#include "Algorithm.h"


void ImageAlgorithm::GetIntegralImage(Mat img,double **integralImage)
{
	int channels = img.channels();
	int imageCols = img.cols;
	int imageRows = img.rows;
	/*积分图的第一行和第一列都等于原始图像第一行，第一列的累加*/
	/*遍历每一列的第一个元素*/
	for (int i = 0; i < imageCols; ++i) {
		for (int m = 0; m < channels; ++m) {
			if(channels==1)integralImage[m][i] = (double)img.at<uchar>(0, i);
			else if(channels > 1)integralImage[m][i] = (double)img.at<Vec3b>(0, i)[m];
			if (i >= 1) {
				integralImage[m][i] += integralImage[m][i-1];
			}
			cout << "m:" << m << "," << "i:" << i << "," << integralImage[m][i] << endl;
		}
	}
	/*遍历每一行的第一个元素*/
	for (int i = 1; i < imageRows; ++i) {
		for (int m = 0; m < channels; ++m) {
			if(channels==1)integralImage[m][ i * imageCols] = img.at<uchar>(i, 0) + integralImage[m][(i - 1) * imageCols];
			else if(channels>1)integralImage[m][i * imageCols] = img.at<Vec3b>(i, 0)[m] + integralImage[m][(i - 1) * imageCols];
		}
	}
	/*其他位置的积分图*/
	for (int i = 1; i < imageRows; ++i) {
		for (int j = 1; j < imageCols; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					integralImage[m][j + i * imageCols] = img.at<uchar>(i, j) + integralImage[m][(j-1) + i * imageCols] + integralImage[m][j + (i-1) * imageCols] - integralImage[m][(j-1) + (i-1) * imageCols];
				}
				else if (channels > 1) {
					integralImage[m][j + i * imageCols] = img.at<Vec3b>(i, j)[m] + integralImage[m][(j-1) + i * imageCols] + integralImage[m][j + (i-1) * imageCols] - integralImage[m][(j-1) + (i-1) * imageCols];
				}
			}
		}
	}

}

/*
函数作用：对图像进行边缘填充，防止卷积操作造成图像边缘丢失
函数参数：1、img：需要进行扩充的图像
		 2、size：要扩充的边缘的大小
函数返回值：扩充以后的图像
*/
Mat ImageAlgorithm::imageEdgeExpand(Mat img, int size)
{
	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();

	/*创建一个新的图像，这个图像的大小为原图像的大小加上扩充的边缘大小*/
	Mat imgRes;
	imgRes.create(rows + size * 2, cols + size * 2, img.type());

	/*将原图像复制到新图像的中间部分*/
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels > 1) {
					imgRes.at<Vec3b>(i + size, j + size)[m] = img.at<Vec3b>(i, j)[m];
				}
				else if (channels == 1) {
					imgRes.at<uchar>(i + size, j + size) = img.at<uchar>(i, j);
				}
			}
		}
	}

	/*对图像的边缘进行填充*/
	for (int i = 0; i < size; ++i) {
		/*填充左边缘*/
		for (int j = 0; j < rows + 2 * size; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(j, i) = imgRes.at<uchar>(j, size);
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(j, i)[m] = imgRes.at<Vec3b>(j, size)[m];
				}
			}
		}
		/*填充右边缘*/
		for (int j = 0; j < rows + 2 * size; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(j, cols+size+i) = imgRes.at<uchar>(j, cols + size -1);
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(j, i+cols+size)[m] = imgRes.at<Vec3b>(j, cols+size-1)[m];
				}
			}
		}
		/*填充上边缘*/
		for (int j = 0; j < cols + 2 * size; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(i, j) = imgRes.at<uchar>(size, j);
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(i, j)[m] = imgRes.at<Vec3b>(size, j)[m];
				}
			}
		}
		/*填充下边缘*/
		for (int j = 0; j < cols + 2 * size; ++j) {
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(i + rows + size, j) = imgRes.at<uchar>(rows + size - 1,j);
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(i + rows + size, j)[m] = imgRes.at<Vec3b>(rows + size - 1, j)[m];
				}
			}
		}
	}
	return imgRes;
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
//Mat ImageAlgorithm::imageNoiseAddition(Mat img, int dx, int dy, double Scale_x, double Scale_y, double angle, int choice, int option)
//{
//    /*处理逻辑：根据选择进行图像的基本变换，返回处理后的像素矩阵*/
//    switch (option)
//	{
//	case IMAGE_TRANSLATION :
//		return imageTranslation(img, dx, dy); /*图像平移*/
//	case IMAGE_RESIZING:
//		return imageResizing(img, Scale_x, Scale_y); /*图像缩放*/
//	case IMAGE_ROTATING:
//		return imageRotating(img,/* double img_cols, double img_rows,*/angle); /*图像旋转*/
//	case IMAGE_REFLECTION:
//		return imageReflection(img, choice); /*图像镜像*/
//	default:
//		break;
//	}
//	return img;
//}

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
	//imshow("translatedImage", translatedImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

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
	/*创建一个用于存储缩放后的像素矩阵*/
	Mat resizingImage;
	/*按横向和纵向比例缩放图像*/
	resize(img, resizingImage, Size(), Scale_x, Scale_y);
	/*创建一个窗口显示图像*/
	//imshow("resizingImage", resizingImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

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
	//imshow("rotatingImage", rotatingImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

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
	//imshow("reflectionImage", reflectionImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return reflectionImage;
}

/*
函数作用：图像变灰度处理函数，根据option的不同，选择不同的变换方法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		  2、option：1：输出灰度图像，2：输出2值图像
返回值：返回经过图像变灰度后的像素矩阵
*/
Mat ImageAlgorithm::imageGray(Mat img, int option)
{
	/*处理逻辑：根据选择变灰度函数，返回处理后的像素矩阵*/
	switch (option)
	{
	case IMAGE_GRAYSCALE:
		return imageGrayScale(img); /*图像变为灰度图像*/
	case IMAGE_GRAYBINARY:
		return imageGrayBinary(img);/*图像变为2值图像*/
	default:
		break;
	}
	return img;
}

/*
函数作用：灰度图像函数，实现将彩色图像变为灰度图像
函数参数：1、img：传入的像素矩阵
返回值：返回图像变灰度后的像素矩阵
*/
Mat ImageAlgorithm::imageGrayScale(Mat img)
{
	/*创建用于存储图像变灰度后的像素矩阵*/
	Mat grayScaleImage;
	/*图像变灰度图像*/
	cvtColor(img, grayScaleImage, COLOR_BGR2GRAY);
	/*创建一个窗口显示图像*/
	//imshow("grayScaleImage", grayScaleImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return grayScaleImage;
}

/*
函数作用：2值图像函数，实现将彩色图像变为2值图像
函数参数：1、img：传入的像素矩阵
返回值：返回图像变2值图像后的像素矩阵
*/
Mat ImageAlgorithm::imageGrayBinary(Mat img)
{
	/*创建用于存储图像变灰度后的像素矩阵*/
	Mat grayBinaryImage;
	/*灰度像素矩阵*/
	Mat grayScaleImage;
	/*图像变2值图像*/
	/*先将其变为灰度矩阵*/
	cvtColor(img, grayScaleImage, COLOR_BGR2GRAY);
	/*再将灰度矩阵变为2值矩阵，将图像中的像素值与阈值（此处为128）进行比较，
	并根据比较结果将像素值设置为两个给定的输出值（此处为0和255）*/
	threshold(grayScaleImage, grayBinaryImage, 128, 255, THRESH_BINARY);
	/*创建一个窗口显示图像*/
	//imshow("grayBinaryImage", grayBinaryImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return grayBinaryImage;
}

/*
函数作用：钝化边缘函数，实现图像的边缘钝化
函数参数：1、img：传入的像素矩阵
返回值：返回图像的边缘钝化后的像素矩阵
*/
Mat ImageAlgorithm::imageBlurring(Mat img)
{
	/*图像变灰度图像*/
	Mat grayScaleImage;
	cvtColor(img, grayScaleImage, COLOR_BGR2GRAY);

	/*对灰度图像进行高斯模糊处理，以减少噪声*/
	Mat blurredImage;
	/*Size内的参数为高斯核的大小，参数0表示默认高斯核在X和Y方向上的标准差相同*/
	GaussianBlur(grayScaleImage, blurredImage, Size(5, 5), 0);

	/*进行拉普拉斯边缘检测，以检测图像中的边缘信息*/
	Mat laplacianImage;
	/*CV_8U表示输出图像的深度，3表示拉普拉斯核的大小*/
	Laplacian(blurredImage, laplacianImage, CV_8U, 3);

	/*创建用于存储图像的边缘钝化后的像素矩阵*/
	Mat blurringImage;
	/*将原始灰度图像减去拉普拉斯边缘图像，实现图像的边缘钝化*/
	blurringImage = grayScaleImage - laplacianImage;

	/*创建一个窗口显示图像*/
	//imshow("blurringImage", blurringImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return blurringImage;
}

/*
函数作用：锐化边缘函数，实现图像的边缘锐化
函数参数：1、img：传入的像素矩阵
返回值：返回图像的边缘锐化后的像素矩阵
*/
Mat ImageAlgorithm::imageSharpening(Mat img)
{
	/*图像变灰度图像*/
	Mat grayScaleImage;
	cvtColor(img, grayScaleImage, COLOR_BGR2GRAY);

	/*对灰度图像进行高斯模糊处理，以减少噪声*/
	Mat blurredImage;
	/*Size内的参数为高斯核的大小，参数0表示默认高斯核在X和Y方向上的标准差相同*/
	GaussianBlur(grayScaleImage, blurredImage, Size(5, 5), 0);

	/*进行拉普拉斯边缘检测，以检测图像中的边缘信息*/
	Mat laplacianImage;
	/*CV_8U表示输出图像的深度，3表示拉普拉斯核的大小*/
	Laplacian(blurredImage, laplacianImage, CV_8U, 3);

	/*创建用于存储图像的边缘锐化后的像素矩阵*/
	Mat sharpeningImage;
	/*将原始灰度图像与拉普拉斯边缘图像进行加权相加，实现图像的边缘锐化*/
	/*参数1.5表示第一个图像（即灰度图像）的权重，参数-0.5表示第二个图像（即拉普拉斯边缘检测后的图像）的权重，
	  参数0用于调整结果图像的亮度*/
	addWeighted(grayScaleImage, 1.5, laplacianImage, -0.5, 0, sharpeningImage);

	///*创建用于存储图像的边缘钝化后的像素矩阵*/
	//Mat blurringImage;
	///*将原始灰度图像减去拉普拉斯边缘图像，实现图像的边缘钝化*/
	//blurringImage = grayScaleImage - laplacianImage;

	/*创建一个窗口显示图像*/
	//imshow("sharpeningImage", sharpeningImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return sharpeningImage;
}

/*
函数作用：图像加噪声处理函数，根据option的不同，选择不同的加噪方式
函数参数：1、img：传入的需要处理的图像的像素矩阵
		  2、option：1：加高斯噪声，2：加椒盐噪声，3：加泊松噪声
返回值：返回加噪声后的的像素矩阵
*/
Mat ImageAlgorithm::imageAddNoise(Mat img, int option)
{
	/*处理逻辑：根据选择加噪函数，返回处理后的像素矩阵*/
	switch (option)
	{
	case GAUSSIANNOISE:
		return imageGaussianNoise(img);  /*加高斯噪声*/
	case SALTPEPPERNOISE:
		return imageSaltPepperNoise(img);/*加椒盐噪声*/
	case POISSONNOISE:
		return imagePoissonNoise(img);   /*加泊松噪声*/
	default:
		break;
	}
	return img;
}

/*
函数作用：高斯噪声函数，实现给图像加高斯噪声
函数参数：1、img：传入的像素矩阵
返回值：返回加高斯噪声后上的像素矩阵
*/
Mat ImageAlgorithm::imageGaussianNoise(Mat img)
{
	/*创建一个用于存储加高斯噪声后的像素矩阵*/
	Mat gaussiannoiseImage;

	/*设置高斯噪声参数*/
	double mean = 0.0;    //均值
	double stddev = 30.0; //标准差
	/*生成高斯噪声图像*/
	Mat noise(img.size(), CV_32FC3); //创建一个与原图像相同尺寸、每个像素由3个32位浮点数组成的噪声图像
	//randn(noise, mean, stddev); //生成服从均值为mean，标准差为stddev的高斯分布的随机数，并存储到 noise 矩阵中
	randn(noise, Scalar::all(mean), Scalar::all(stddev));//与前一个的区别在于使用了Scalar对象来表示均值和标准差，可以方便地将相同的值应用于每个通道
	/*img图像将被转换为每个像素由3个32位浮点数组成的图像，并且结果将存储在 gaussiannoiseImage 矩阵中，
	  此转换在添加高斯噪声前进行，确保能够正确处理和操作浮点数值*/
	img.convertTo(gaussiannoiseImage, CV_32FC3);
	/*将噪声矩阵 noise 的值逐个地加到图像矩阵 gaussiannoiseImage 对应上（即将每个像素的颜色值与对应位置上的噪声值相加，从而在图像上添加噪声）*/
	gaussiannoiseImage += noise;
	/*将 gaussiannoiseImage 矩阵的数据类型将从 CV_32FC3 转换为每个像素由3个8位无符号整数数组成的图像*/
	gaussiannoiseImage.convertTo(gaussiannoiseImage, CV_8UC3);

	// 添加高斯噪声到图像
	//add(img, noise, gaussiannoiseImage, Mat(), CV_8UC3);

	/*创建一个窗口显示原图像*/
	//imshow("Image", img);
	/*创建一个窗口显示加高斯噪声后的图像*/
	//imshow("gaussiannoiseImage", gaussiannoiseImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return gaussiannoiseImage;
}

/*
函数作用：椒盐噪声函数，实现给图像加椒盐噪声
函数参数：1、img：传入的像素矩阵
返回值：返回加椒盐噪声后上的像素矩阵
*/
Mat ImageAlgorithm::imageSaltPepperNoise(Mat img)
{
	/*创建一个用于存储加椒盐噪声后的像素矩阵*/
	Mat saltpeppernoiseImage = img.clone();

	/*添加椒盐噪声*/ 
	float noise_ratio = 0.02; //噪声比例
	/*通过像素矩阵的行数和列数来计算噪声像素的数量*/
	int num_noise_pixels = saltpeppernoiseImage.rows * saltpeppernoiseImage.cols * noise_ratio;
	/*通过随机选择像素位置，并将其设置为黑色（椒噪声）或白色（盐噪声）来模拟椒盐噪声的效果*/
	for (int i = 0; i < num_noise_pixels; i++) {
		/*通过行数和列数分别生成随机的行数，列数*/
		int row = rand() % saltpeppernoiseImage.rows;
		int col = rand() % saltpeppernoiseImage.cols;

		if (rand() % 2 == 0) {
			/*将像素设置为黑色表示椒噪声(pepper noise)*/
			saltpeppernoiseImage.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
		}
		else {
			/*将像素设置为白色表示盐噪声(salt noise)*/
			saltpeppernoiseImage.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
		}
	}

	/*创建一个窗口显示原图像*/
	//imshow("Image", img);
	/*创建一个窗口显示加高斯噪声后的图像*/
	//imshow("saltpeppernoiseImage", saltpeppernoiseImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return saltpeppernoiseImage;
}

/*
函数作用：泊松噪声函数，实现给图像加泊松噪声
函数参数：1、img：传入的像素矩阵
返回值：返回加泊松噪声后上的像素矩阵
*/
Mat ImageAlgorithm::imagePoissonNoise(Mat img)
{
	/*创建一个用于存储加泊松噪声后的像素矩阵*/
	Mat poissonnoiseImage;

	/*生成泊松噪声图像*/
	Mat noise(img.size(), CV_32FC3);
	/*生成服从均值为0、标准差为16的正态分布的随机数存于 noise 矩阵中*/
	randn(noise, Scalar(0, 0, 0), Scalar(16, 16, 16));
	/*img图像将被转换为每个像素由3个32位浮点数组成的图像，并且结果将存储在 poissonnoiseImage 矩阵中*/
	img.convertTo(poissonnoiseImage, CV_32FC3);
	/*将噪声矩阵 noise 的值逐个地加到图像矩阵 poissonnoiseImage 对应上（即将每个像素的颜色值与对应位置上的噪声值相加，从而在图像上添加噪声）*/
	poissonnoiseImage += noise;
	/*将 poissonnoiseImage 矩阵的数据类型将从 CV_32FC3 转换为每个像素由3个8位无符号整数数组成的图像*/
	poissonnoiseImage.convertTo(poissonnoiseImage, CV_8UC3);
	
	/*创建一个窗口显示原图像*/
	//imshow("Image", img);
	/*创建一个窗口显示加高斯噪声后的图像*/
	//imshow("poissonnoiseImage", poissonnoiseImage);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return poissonnoiseImage;
}

/*
函数作用：图像直方图绘制函数，实现图像的直方图绘制
函数参数：1、img：传入的像素矩阵
返回值：返回直方图的图像矩阵
*/
Mat ImageAlgorithm::imageHistogram(Mat img)
{
	/*将图像转换为HSV颜色空间*/
	Mat hsvImage;
	cvtColor(img, hsvImage, COLOR_BGR2HSV);
	/*分割HSV图像的通道*/
	vector<Mat> channels;
	split(hsvImage, channels);

	/*灰度图像的直方图实现:
    //图像变灰度图像
	Mat grayScaleImage;
	cvtColor(img, grayScaleImage, COLOR_BGR2GRAY);
	//计算图像的直方图
	int histSize = 256; //直方图尺寸
	float range[] = { 0, 256 }; //像素值范围
	const float* histRange = { range };
	//bool uniform = true; //直方图是否均匀
	//bool accumulate = false; //直方图是否累积
	
	Mat hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	//创建直方图窗口并绘制直方图
	int histWidth = 512;
	int histHeight = 400;
	int binWidth = cvRound((double)histWidth / histSize);
    //创建一个用于绘制直方图的图像矩阵
	Mat histogramImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));

	normalize(hist, hist, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histogramImage, Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
			Point(binWidth * (i), histHeight - cvRound(hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}*/
	
	/*计算每个通道的直方图*/
	int histSize = 256;         //直方图尺寸
	float range[] = { 0, 256 }; //像素值范围
	const float* histRange = { range };
	//bool uniform = true;        //直方图是否均匀
	//bool accumulate = false;    //直方图是否累积
	/*使用calcHist函数计算图像的直方图*/
	Mat hist_h, hist_s, hist_v;
	/*参数依次为：输入图像的数组（可以是多个图像，每个图像是一个单通道图像），输入图像的数量，
	指定要计算直方图的通道（此处为单通道图像，故为0），用于指定感兴趣区域（ROI，即在指定的区域内的像素才会被用于计算直方图），
	输出的直方图（一个单通道矩阵），直方图的维度，直方图的尺寸（即每个维度的直方图的bin数量），每个维度的像素值范围*/
	calcHist(&channels[0], 1, 0, Mat(), hist_h, 1, &histSize, &histRange/*, uniform, accumulate*/);
	calcHist(&channels[1], 1, 0, Mat(), hist_s, 1, &histSize, &histRange/*, uniform, accumulate*/);
	calcHist(&channels[2], 1, 0, Mat(), hist_v, 1, &histSize, &histRange/*, uniform, accumulate*/);


	/*创建直方图窗口并绘制直方图*/
	int histWidth = 512;  //直方图图像的高度（像素数）
	int histHeight = 400; //直方图图像的宽度（像素数）
	int binWidth = cvRound((double)histWidth / histSize);
	/*创建一个用于绘制直方图的图像矩阵作为直方图的背景，在其上绘制直方图的线条*/
	/*CV_8UC3：指定图像矩阵的数据类型为8位无符号整数，通道数为3（表示彩色图像）,
	  Scalar(0, 0, 0)：指定图像矩阵的初始值为白色（BGR通道的值分别为0）*/
	Mat histogramImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));
	/*对直方图进行归一化,以便直方图能够适应绘制直方图的图像尺寸,
	  归一化后的直方图将被用于绘制直方图的线条*/
	/*参数分别为：要归一化的直方图，归一化后的直方图，归一化的最小值，归一化的最大值（即归一化后的直方图的最大高度），
	  归一化的类型（这里使用最小-最大归一化），归一化的范围（默认为输入图像的全局范围），用于计算直方图的掩码（这里不使用掩码）*/
	normalize(hist_h, hist_h, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());
	normalize(hist_s, hist_s, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());
	normalize(hist_v, hist_v, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());
	/*绘制直方图的线条*/
	for (int i = 1; i < histSize; i++)
	{
		/*line函数参数分别为：直方图图像矩阵，线条的起始点（即当前bin的前一个bin的位置），线条的终点（即当前bin的位置），
		  线条的颜色（以下分别代表蓝色，绿色，红色），线条的宽度，线条的连接类型，线条的偏移*/
		line(histogramImage, Point(binWidth * (i - 1), histHeight - cvRound(hist_h.at<float>(i - 1))),
			Point(binWidth * (i), histHeight - cvRound(hist_h.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);

		line(histogramImage, Point(binWidth * (i - 1), histHeight - cvRound(hist_s.at<float>(i - 1))),
			Point(binWidth * (i), histHeight - cvRound(hist_s.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);

		line(histogramImage, Point(binWidth * (i - 1), histHeight - cvRound(hist_v.at<float>(i - 1))),
			Point(binWidth * (i), histHeight - cvRound(hist_v.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	
	/*创建一个直方图窗口，并在其中绘制直方图*/
	//imshow("histogramImage", histogramImage);
	/*创建一个窗口显示图像*/
	//imshow("Image", img);
	/*图像显示的时间，为系统结束前的阻塞时间，如果想要看到图片显示效果，建议此值设置在（3000以上，单位ms）*/
	//waitKey(0);

	return histogramImage;
}

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
Mat ImageAlgorithm::imageDenoising(Mat img, int kernel_size,int option,int level)
{
	switch (option)
	{
	case AVERAGE_FILTER:
		return imageAverageFilter(img, kernel_size);
	case MEDIAN_FILTER:
		return imageMedianFilter(img, kernel_size);
	case GAUSSIAN_FILTER:
		return imageGaussianFilter(img, kernel_size);
	case BILATERAL_FILTER:
		return imageBilateralFilter(img, kernel_size);
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
返回值：返回经过均值滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageAverageFilter(Mat img, int kernel_size)
{	
	int channels = img.channels();
	int imageRows = img.rows;
	int imageCols = img.cols;
	Mat imgRes;
	/*初始化图像结果图*/
	imgRes.create(img.size(), img.type());
	/*动态分配sum数组的空间大小*/
	double*sum = new double[channels];
	for (int i = 0; i < channels; ++i) {
		sum[i] = 0;
	}
	/*定义积分图*/
	double** integralImage = new double* [channels];
	int data_num = imageRows * imageCols;
	for (int i = 0; i < channels; ++i) {
		integralImage[i] = new double[data_num];
	}
	/*获取积分图*/
	GetIntegralImage(img,integralImage);
	/*遍历图像所有像素，通过积分图进行均值处理*/
	for (int i = 0; i < imageRows; ++i) {
		for (int j = 0; j < imageCols; ++j) {
			/*计算滤波窗口左上角和右下角的坐标*/
			int top = max(0, i - kernel_size / 2);
			int left = max(0, j - kernel_size / 2);
			int bottom = min(imageRows - 1, i + kernel_size / 2);
			int right = min(imageCols - 1, j + kernel_size / 2);
			int count = (bottom - top + 1) * (right - left + 1);
			for (int m = 0; m < channels; ++m) {
				/*窗口内的和就等于，右下角+左上角(x-1,y-1)-右上角(x-1)-左下角(y-1)*/
				sum[m] = integralImage[m][bottom * imageCols + right];
				if (top > 0) sum[m] -= integralImage[m][(top - 1) * imageCols + right];
				if (left > 0) sum[m] -= integralImage[m][bottom * imageCols +left-1];
				if (top > 0 && left > 0) sum[m] += integralImage[m][(top - 1) * imageCols + (left - 1)];
				if (channels == 1) {
					imgRes.at<uchar>(i, j) = saturate_cast<uchar>(sum[m] / count);
				}
				else {
					imgRes.at<Vec3b>(i, j)[m] = saturate_cast<uchar>(sum[m] / count);
				}
			}
		}
	}
	delete []sum;
	for (int i = 0; i < channels; ++i) {
		delete[] integralImage[i];
		integralImage[i] = NULL;
	}
	delete[] integralImage;
	integralImage = NULL;
	return imgRes;
}

/*
函数作用：中值滤波处理图像，去除椒盐噪声
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：窗口大小
返回值：返回经过中值滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageMedianFilter(Mat img, int kernel_size)
{
	/*窗口的半径*/
	int r = kernel_size / 2;
	/*对原图进行边缘扩充*/
	Mat newImage;
	newImage = imageEdgeExpand(img, r);
	int channels = img.channels();
	int imageRows = img.rows;
	int imageCols = img.cols;
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*根据通道数量，窗口大小，动态的开辟存储窗口灰度的Mat，用于进行灰度统计和中值选取*/
	Mat window;
	window.create(kernel_size, kernel_size, newImage.type());
	/*动态开辟内存空间，存储灰度直方图*/
	int** hist = new int* [channels];
	for (int i = 0; i < channels; ++i) {
		hist[i] = new int[256];
	}
	/*x，y这两层循环用于遍历图像的所有像素*/
	for (int x = 0; x < imageRows; ++x) {
		for (int y = 0; y < imageCols; ++y) {
			/*i，j这两层循环用于遍历窗口中的元素*/
			for (int i = -r; i <= r; ++i) {
				for (int j = -r; j <= r; ++j) {
					/*m这层循环是用于遍历通道数，一般彩色图片都是RGB3通道，如果传进来的是灰度图像，那么channels=1*/
					for (int m = 0; m < channels; ++m) {
						/*单通道使用单通道的读取方式，多通道使用多通道的读取方式*/
						if (channels == 1)window.at<uchar>(i+r,j+r) = newImage.at<uchar>(x+i+r, y+j+r);
						else if (channels > 1) window.at<Vec3b>(i + r, j + r)[m] = newImage.at<Vec3b>(x+i+r, y+j+r)[m];
					} 
				}
			}
			/*灰度统计*/
			imgageStatisticalHistogram(window, hist);
			/*计算灰度图像中的中值*/
			for (int m = 0; m < channels; ++m) {
				int sum = 0;
				for (int i = 0; i < 256; ++i) {
					sum += hist[m][i];
					/*此时这个灰度值i就是所有灰度值的中间值*/
					if (sum >= kernel_size * kernel_size / 2) {
						if (channels == 1)imgRes.at<uchar>(x, y) = saturate_cast<uchar>(i);
						else if (channels > 1) imgRes.at<Vec3b>(x, y)[m] = saturate_cast<uchar>(i);
						break;
					}
				}
			}
		}
	}

	/*释放资源*/
	for (int i = 0; i < channels; ++i) {
		delete[] hist[i];
	}
	delete[] hist;
	return imgRes;
}

/*
函数作用：高斯滤波处理图像，对图像平滑处理最好的方法
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：卷积核大小
返回值：返回经过高斯滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageGaussianFilter(Mat img, int kernel_size)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*窗口半径*/
	int r = kernel_size / 2;
	Mat newImage;
	/*对图像进行边缘扩充，防止边缘丢失*/
	newImage = imageEdgeExpand(img, r);
	/*图片通道数*/
	int channels = img.channels();
	/*根据窗口大小和通道数量动态分配窗口内存空间*/
	double* sum = new double[channels];
	/*灰度值的数量*/
	int gray_num = 256;
	/*生成一维高斯滤波的存储空间*/
	double* kernel = new double[kernel_size];
	/*平均差sigma*/
	double sigma = kernel_size / 6.0;
	/*计算一维高斯滤波*/
	double tmp = 0;
	for (int i = 0; i < kernel_size; ++i) {
		double g = exp(-pow(2,i-r) / (2*sigma*sigma));
		tmp += g;
		kernel[i] = g;
	}
	/*归一化，去除常数*/
	for (int i = 0; i < kernel_size; ++i) {
		kernel[i] /= tmp;
	}
	/*创建乘法表,一共需要进行kernel_size*256次乘法，求出所有可能性，模板越大，节省时间越多*/
	double* mulitiplicationTable = new double[gray_num * kernel_size];
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < gray_num; ++j) {
			mulitiplicationTable[i * gray_num + j] = kernel[i] * j;
		}
	}
	
	/*得到了高斯掩膜和乘法表，现在开始进行高斯滤波*/
	/*进行水平滤波处理*/
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < channels; ++m) sum[m] = 0.0;
			for (int x = -r; x <= r; ++x) {
				for (int m = 0; m < channels; ++m) {
					if (channels == 1) {
						/*kernel中参数的位置*256+这个像素点的灰度值，就是它们在乘法表中的值*/
						sum[m] += mulitiplicationTable[(x + r) * gray_num + newImage.at<uchar>(i, j + x + r)];
					}
					else if (channels > 1) {
						sum[m] += mulitiplicationTable[(x + r) * gray_num + newImage.at<Vec3b>(i, j + x + r)[m]];
					}
				}
			}
			/*将值赋给imgRes，同时进行阈值控制*/
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(i, j) = saturate_cast<uchar>(sum[m] > 255 ? 255 : (sum[m] < 0 ? 0 : sum[m]));
					newImage.at<uchar>(i, j + r) = imgRes.at<uchar>(i, j);
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(i, j)[m] = saturate_cast<uchar>(sum[m] > 255 ? 255 : (sum[m] < 0 ? 0 : sum[m]));
					newImage.at<Vec3b>(i, j + r)[m] = imgRes.at<Vec3b>(i, j)[m];
				}
			}
		}
	}

	/*进行垂直滤波处理*/
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			/*进行水平滤波处理*/
			for (int m = 0; m < channels; ++m) sum[m] = 0.0;
			for (int x = -r; x <= r; ++x) {
				for (int m = 0; m < channels; ++m) {
					if (channels == 1) {
						/*kernel中参数的位置*256+这个像素点的灰度值，就是它们在乘法表中的值*/
						sum[m] += mulitiplicationTable[(x + r) * gray_num + newImage.at<uchar>(i+x+r, j)];
					}
					else if (channels > 1) {
						sum[m] += mulitiplicationTable[(x + r) * gray_num + newImage.at<Vec3b>(i+x+r,j)[m]];
					}
				}
			}
			/*将值赋给imgRes，同时进行阈值控制*/
			for (int m = 0; m < channels; ++m) {
				if (channels == 1) {
					imgRes.at<uchar>(i, j) = saturate_cast<uchar>(sum[m] > 255 ? 255 : (sum[m] < 0 ? 0 : sum[m]));
				}
				else if (channels > 1) {
					imgRes.at<Vec3b>(i, j)[m] = saturate_cast<uchar>(sum[m] > 255 ? 255 : (sum[m] < 0 ? 0 : sum[m]));
				}
			}
		}
	}
	delete[] sum;
	delete[] kernel;
	return imgRes;
}

/*
函数作用：双边滤波处理图像，可用于图像降噪，边缘保留等
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、kernel_size：卷积核大小
		 3、space_sigma:空间平均差
		 4、color_sigma：灰度平均差
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageBilateralFilter(Mat img, int kernel_size,double space_sigma, double color_sigma)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	/*空间权重和灰度值权重的sigma的平方*/
	double space_coffe = -0.5 / (space_sigma * space_sigma);
	double color_coffe = -0.5 / (color_sigma * color_sigma);
	/*窗口半径*/
	int r = kernel_size / 2;
	/*进行边缘扩充*/
	Mat newImage;
	newImage = imageEdgeExpand(img, r);
	/*灰度数量*/
	int gray_num = 256;
	/*图片通道数*/
	int channels = img.channels();
	/*通道灰度值的求和*/
	double* sum = new double[channels];
	/*通道权重的求和*/
	double* weight_sum = new double[channels];
	for (int i = 0; i < channels; ++i) {
		sum[i] = 0;
		weight_sum[i] = 0;
	}
	/*计算空间权重*/
	double* space_weight = new double[(kernel_size * kernel_size)];
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			double tmp = sqrt((i - r) * (i - r) + (j - r) * (j - r));
			space_weight[i * kernel_size + j] = exp(tmp*tmp*space_coffe);
		}
	}
	/*我们可以确定，计算值域比重时，结果范围在0-255之间，那么我们可以提前计算出值域比重*/
	double* color_weight = new double[gray_num];
	for (int i = 0; i < gray_num; ++i) {
		color_weight[i] = exp( i*i * color_coffe);
	}

	/*x,y循环遍历图像的所有像素*/
	for (int x = 0; x < img.rows; ++x) {
		for (int y = 0; y < img.cols; ++y) {
			/*i,j循环遍历窗口*/
			for (int i = -r; i <= r; ++i) {
				for (int j = -r; j <= r; ++j) {
					/*m循环遍历通道*/
					for (int m = 0; m < channels; ++m) {
						if (channels == 1) {
							/*空间权重*/
							double space = space_weight[(i+r) * kernel_size + (j+r)];
							/*灰度值权重*/
							double color = color_weight[abs(newImage.at<uchar>(x + i + r, y + j + r) - newImage.at<uchar>(x+r, y+r))];
							/*双边滤波权重*/
							double weight = color * space;
							/*对权重求和的操作，便于进行归一化处理*/
							sum[m] += newImage.at<uchar>(x+i+r,y+j+r) * weight;
							/*对结果进行求和*/
							weight_sum[m] += weight;
						}
						else if (channels > 1) {
							/*空间权重*/
							double space = space_weight[(i+r) * kernel_size + (j+r)];
							/*灰度值权重*/
							double color = color_weight[abs(newImage.at<Vec3b>(x + i+r, y + j+r)[m] - newImage.at<Vec3b>(x+r, y+r)[m])];
							/*双边滤波权重*/
							double weight = color * space;
							/*对权重求和的操作，便于进行归一化处理*/
							sum[m] += newImage.at<Vec3b>(x + i+r, y + j+r)[m] * weight;
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
	delete [] space_weight;
	delete[] color_weight;
	return imgRes;
}


// 一维离散小波变换
void waveletTransform1D(vector<double>& input, vector<double>& output)
{
	int N = input.size();
	output.resize(N);

	int h = N / 2;
	for (int i = 0; i < h; i++) {
		output[i] = (input[2 * i] + input[2 * i + 1]) / 2;
		output[i + h] = (input[2 * i] - input[2 * i + 1]) / 2;
	}
}

// 一维离散小波逆变换
void inverseWaveletTransform1D(vector<double>& input, vector<double>& output)
{
	int N = input.size();
	output.resize(N);

	int h = N / 2;
	for (int i = 0; i < h; i++) {
		output[2 * i] = input[i] + input[i + h];
		output[2 * i + 1] = input[i] - input[i + h];
	}
}
// 二维离散小波变换
void waveletTransform2D(Mat& input, Mat& output, int levels)
{
	Mat temp = input.clone();

	for (int level = 0; level < levels; level++) {
		// 对每一行进行小波变换
		for (int i = 0; i < temp.rows; i++) {
			vector<double> rowInput, rowOutput;
			for (int j = 0; j < temp.cols; j++) {
				rowInput.push_back(temp.at<double>(i, j));
			}
			waveletTransform1D(rowInput, rowOutput);
			for (int j = 0; j < temp.cols; j++) {
				temp.at<double>(i, j) = rowOutput[j];
			}
		}

		// 对每一列进行小波变换
		for (int j = 0; j < temp.cols; j++) {
			vector<double> colInput, colOutput;
			for (int i = 0; i < temp.rows; i++) {
				colInput.push_back(temp.at<double>(i, j));
			}
			waveletTransform1D(colInput, colOutput);
			for (int i = 0; i < temp.rows; i++) {
				temp.at<double>(i, j) = colOutput[i];
			}
		}

		// 缩小图像尺寸
		resize(temp, output, Size(temp.cols / 2, temp.rows / 2));

		temp = output.clone();
	}
}
// 二维离散小波逆变换
void inverseWaveletTransform2D(Mat& input, Mat& output, int levels)
{
	Mat temp = input.clone();

	for (int level = levels - 1; level >= 0; level--) {
		// 还原图像尺寸
		Mat temp2;
		resize(temp, temp2, Size(temp.cols * 2, temp.rows * 2));

		// 对每一列进行小波逆变换
		for (int j = 0; j < temp2.cols; j++) {
			vector<double> colInput, colOutput;
			for (int i = 0; i < temp2.rows; i++) {
				colInput.push_back(temp2.at<double>(i, j));
			}
			inverseWaveletTransform1D(colInput, colOutput);
			for (int i = 0; i < temp2.rows; i++) {
				temp2.at<double>(i, j) = colOutput[i];
			}
		}

		// 对每一行进行小波逆变换
		for (int i = 0; i < temp2.rows; i++) {
			vector<double> rowInput, rowOutput;
			for (int j = 0; j < temp2.cols; j++) {
				rowInput.push_back(temp2.at<double>(i, j));
			}
			inverseWaveletTransform1D(rowInput, rowOutput);
			for (int j = 0; j < temp2.cols; j++) {
				temp2.at<double>(i, j) = rowOutput[j];
			}
		}

		temp = temp2.clone();
	}

	output = temp.clone();
}
// 对小波系数进行阈值处理
void thresholding(Mat& input, double threshold)
{
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<double>(i, j) < threshold) {
				input.at<double>(i, j) = 0;
			}
		}
	}
}
// 灰度图像小波滤波
void waveletFilterGray(Mat& input, Mat& output, int levels, double threshold)
{
	// 将灰度图像转为双精度浮点型
	Mat inputDouble;
	input.convertTo(inputDouble, CV_64FC1);

	// 进行小波变换
	waveletTransform2D(inputDouble, output, levels);

	// 对小波系数进行阈值处理
	thresholding(output, threshold);

	// 进行小波逆变换
	inverseWaveletTransform2D(output, output, levels);

	// 将结果转为灰度图像
	output.convertTo(output, CV_8UC1);
}

// 彩色图像小波滤波
void waveletFilterColor(Mat& input, Mat& output, int levels, double threshold)
{
	// 将彩色图像转为浮点型
	Mat inputDouble;
	input.convertTo(inputDouble, CV_64FC3);

	// 分离通道
	vector<Mat> channels;
	split(inputDouble, channels);

	// 对每个通道进行小波滤波
	for (int i = 0; i < 3; i++) {
		waveletFilterGray(channels[i], channels[i], levels, threshold);
	}

	// 合并通道
	merge(channels, output);

	// 将结果转为彩色图像
	output.convertTo(output, CV_8UC3);
}

/*
函数作用：小波滤波处理图像，用于信号处理和图像处理中的降噪、压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 2、level：小波去噪的层数
返回值：返回经过双边滤波处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageWaveletFilter(Mat img, int level)
{
	// 灰度图像小波滤波
	Mat filteredImageGray;
	Mat filteredImageColor;
	if(img.channels()==1) waveletFilterGray(img, filteredImageGray, level, 10);
	else if (img.channels() == 3) waveletFilterColor(img, filteredImageColor, level, 10);

	if (img.channels() == 1)return filteredImageGray;
	else return filteredImageColor;
}

/*
函数作用：进行图片的边缘提取，可用于处理物体检测和跟踪，图像分割，图像增强，噪声去噪，图像压缩等领域。
函数参数：1、img：传入的需要处理的图像的像素矩阵
		 3、option：一阶：1:Roberts算子，2:Sobel算子，3:Prewitt算子，4:Kirsch算子，5:Robinson算子，二阶：6:Laplacion算子，7:Canny算子
返回值：返回经过所选择的边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageEdgeDetection(Mat img,int option, int threshold)
{
	/*
	处理逻辑：根据option选择对应的边缘检测算子进行处理，返回处理后的像素矩阵
	*/
	switch (option)
	{
	case ROBERTS:
		return imageRoberts(img,threshold);
	case SOBEL:
		return imageSobel(img);
	case PREWITT:
		return imagePrewitt(img);
	case KIRSCH:
		return imageKirsch(img);
	case ROBINSON:
		return imageRobinson(img);
	case LAPLACIAN:
		return imageLaplacian(img);
	case CANNY:
		return imageCanny(img);
	default:
		break;
	}
	

	return Mat();
}
/*
函数作用：Roberts算子对图片进行边缘检测
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回经过Roberts边缘检测算子处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageRoberts(Mat img, int threshold)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);

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
Mat ImageAlgorithm::imageSobel(Mat img,int kernel_size)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);

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
Mat ImageAlgorithm::imagePrewitt(Mat img)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
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
Mat ImageAlgorithm::imageKirsch(Mat img)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
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
Mat ImageAlgorithm::imageRobinson(Mat img)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);
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
Mat ImageAlgorithm::imageLaplacian(Mat img)
{
	/*将彩色图像转化为灰度图*/
	cvtColor(img, img, COLOR_RGB2GRAY);

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
		 3、c：对比度增强，亮度增强，指数增强的参数。
		 4、r：指数增强的参数
返回值：返回经过对应算法处理过后的像素矩阵
*/
Mat ImageAlgorithm::imageEnhance(Mat img,int option,double c,double r)
{
	switch (option)
	{
	case CONTRAST_ENHANCE:
		return imageContrastEnhance(img, c);
	case BRIGHTNESS:
		return imageBrightness(img,c);
	case HISTOGRAME_QUALIZATION:
		return imageHistogramEqualization(img);
	case EXPONENTIAL_TRANSFORM:
		return imageExponentialTransform(img, c, r);
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
	for (int i = 0; i < img.channels(); ++i) {
		for (int j = 0; j < 256; ++j) {
			hist[i][j] = 0;
		}
	}
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int m = 0; m < img.channels(); ++m) {
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
Mat ImageAlgorithm::imageExponentialTransform(Mat img,double c, double r)
{
	Mat imgRes;
	imgRes.create(img.size(), img.type());
	int channels = img.channels();


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
Mat ImageAlgorithm::imageFourierTransform(Mat image)
{
	cvtColor(image, image, cv::COLOR_BGR2GRAY);
	// 进行傅里叶变换
	Mat padded;
	int m = getOptimalDFTSize(image.rows);
	int n = getOptimalDFTSize(image.cols);
	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImage;
	merge(planes, 2, complexImage);
	dft(complexImage, complexImage);

	// 将频谱移到中心
	split(complexImage, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];
	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage);
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	return magnitudeImage;
}
/*
函数作用：图像融合
函数参数：1、img1：需要进行拼接的图片
		 2、img2：需要进行拼接的图片
返回值：返回融合后图像的像素矩阵
*/
Mat ImageAlgorithm::imageSynthesis(Mat img1, Mat img2) 
{
	/*创建ORB对象*/
	Ptr<ORB> orb = ORB::create();
	/*用于存储图像的关键点和描述符*/
	vector<KeyPoint>keyPoints1, keyPoints2;
	Mat descriptors1, descriptors2;

	/*使用orb算法检测和计算图像的特征点*/
	orb->detectAndCompute(img1,cv::noArray(), keyPoints1, descriptors1);
	orb->detectAndCompute(img2, cv::noArray(), keyPoints2, descriptors2);

	/*匹配特征点*/
	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	/*根据匹配结果筛选出好的匹配点*/
	double min_dist = min_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch& m2) {
		return m1.distance < m2.distance;
		})->distance;
	vector<DMatch> good_matches;
	for (const DMatch& match : matches) {
		if (match.distance <= max(2 * min_dist, 30.0)) {
			good_matches.push_back(match);
		}
	}
	/*使用筛选出来的匹配点进行图像拼接*/
	vector<Point2f> src_pts;
	vector<Point2f> dst_pts;

	for (const DMatch& match : good_matches) {
		src_pts.push_back(keyPoints1[match.queryIdx].pt);
		dst_pts.push_back(keyPoints2[match.trainIdx].pt);
	}

	Mat H = findHomography(src_pts, dst_pts, RANSAC);
	Mat result;
	warpPerspective(img1, result, H, Size(img1.cols + img2.cols, img1.rows));
	Mat roi(result, Rect(0, 0, img2.cols, img2.rows));
	img2.copyTo(roi);

	return result;
}

/*
函数作用：图像分割可以用于目标检测，图像识别，图像增强，医学影像分析等
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回分割后的图像
*/
Mat ImageAlgorithm::imageSegmentation(Mat src)
{
	int row = src.rows;
	int col = src.cols;
	//1. 将RGB图像灰度化
	Mat grayImage;
	cvtColor(src, grayImage, COLOR_BGR2GRAY);
	//2. 使用大津法转为二值图，并做形态学闭合操作
	threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//3. 形态学闭操作
	Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
	//4. 距离变换
	distanceTransform(grayImage, grayImage, DIST_L2, DIST_MASK_3, 5);
	//5. 将图像归一化到[0, 1]范围
	normalize(grayImage, grayImage, 0, 1, NORM_MINMAX);
	//6. 将图像取值范围变为8位(0-255)
	grayImage.convertTo(grayImage, CV_8UC1);
	//7. 再使用大津法转为二值图，并做形态学闭合操作
	threshold(grayImage, grayImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
	//8. 使用findContours寻找marks
	vector<vector<Point>> contours;
	findContours(grayImage, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	Mat marks = Mat::zeros(grayImage.size(), CV_32SC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		//static_cast<int>(i+1)是为了分水岭的标记不同，区域1、2、3...这样才能分割
		drawContours(marks, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i + 1)), 2);
	}
	//9. 对原图做形态学的腐蚀操作
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(src, src, MORPH_ERODE, k);
	//10. 调用opencv的分水岭算法
	watershed(src, marks);
	//11. 随机分配颜色
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// 12. 显示
	Mat dst = Mat::zeros(marks.size(), CV_8UC3);
	int index = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			index = marks.at<int>(i, j);
			if (index > 0 && index <= contours.size()) {
				dst.at<Vec3b>(i, j) = colors[index - 1];
			}
			else if (index == -1)
			{
				dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else {
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}
	return dst;
}

/*
函数作用：识别图像中的数字
函数参数：1、img：传入的需要处理的图像的像素矩阵
返回值：返回处理过后的像素矩阵，图像能够呈现被识别出来的数字
*/
Mat ImageAlgorithm::imageDigitalIdentify(Mat src)
{

	//// 加载已经训练好的KNN模型
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
	knn = cv::Algorithm::load<cv::ml::KNearest>("knn_model.xml");
	Mat tmp = src;

	Mat gray_frame, thres_img, blur_img;
	cvtColor(src, gray_frame, COLOR_BGR2GRAY);//对图像进行预处理（图像去燥二值化）
	GaussianBlur(gray_frame, blur_img, Size(3, 3), 3, 3);
	adaptiveThreshold(blur_img, thres_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 151, 10);
	Mat morph_img, tmp2, tmp3;
	Mat kernerl = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(thres_img, morph_img, MORPH_OPEN, kernerl, Point(-1, -1));
	int k = 5;
	// 查找图像中的轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hiearachy;
	findContours(morph_img, contours, hiearachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++) {
		Rect boundingBox = boundingRect(contours[i]);

		float area = contourArea(contours[i]);
		float ckbi = boundingBox.width / boundingBox.height;

		if (ckbi < 4 && area>50) {
			// 调整数字图像的大小
			Mat digit = morph_img(boundingBox);
			resize(digit, digit, Size(28, 28));

			// 对数字图像进行预处理，与训练数据保持一致
			digit = digit.reshape(0, 1);
			digit.convertTo(digit, CV_32FC1);
			Mat sample = digit(Range(0, digit.rows), Range::all());

			// 使用KNN模型进行预测
			int response = knn->predict(sample);

			// 在原图上绘制分割出的数字
			rectangle(tmp, boundingBox, Scalar(0, 255, 0),1, 8);
			putText(tmp, to_string(response), boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 4);
		}
	}

	return tmp;
}


/*
函数作用：图像转素描
*/
Mat ImageAlgorithm::imageSketch(Mat src)
{
	Mat gray, gray_inverse, dst;
	
	/*图像转灰度图*/
	cvtColor(src, gray, COLOR_BGRA2GRAY);

	//2.图像取反,三种取反的方法
	//2.1 遍历像素直接用255去减
	//gray_inverse = Scalar(255, 255, 255) - gray;
	//2.2 用subtract函数
	//subtract(Scalar(255, 255, 255), gray, gray_inverse);
	//2.3 位运算直接取反
	gray_inverse = ~gray;

	//3 高斯模糊
	GaussianBlur(gray_inverse, gray_inverse, Size(15, 15), 50, 50);

	//4 颜色减淡混合
	divide(gray, 255 - gray_inverse, dst, 255);
	return dst;
}

