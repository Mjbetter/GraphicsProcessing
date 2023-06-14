#include "GraphicsProcessing.h"
#include <QtWidgets/QApplication>
#include "Algorithm.h"
#include "macro.h"

/*
命名规范：
类名：首字母大写，单词和单词之间首字母大写
函数名：变量名称 首字母小写，单词和单词之间首字母大写
*/

/*
注释规范：
注释统一当前格式来包括
注明函数作用
注明参数作用及意义
注明函数返回值
*/

/*
关于算法实现的封装，统一在Algorithm.h中声明函数，在Algorithm.cpp中实现函数
关于QT界面的封装，在Form Files中放置存储的自定义控件，在对应调用位置注释控件相关说明
关于QT界面的代码设置，如果需要用代码对界面进行优化，封装成函数在UI类中实现
关于资源文件的导入，统一放入到GraphicsProcessing.qrc中，资源文件名称统一由英文代替汉字，格式为.png格式
*/

/*
项目每天写完后用git上传至github中，下次在写就拉下来，VS中可以配置git对于项目的快速拉取推送
*/

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GraphicsProcessing w;
    //w.show();

    ImageAlgorithm s;
    Mat img = imread("E:\\数字识别.png");
    //Mat imgx = imread("E:\\3.jpg");
    //Mat imgy = imread("E:\\2.jpg");
    Mat img0,img1, img2, img3, img4, img5;
    // Roberts边缘检测测试
    //img0 = s.imageRoberts(img, NO_FILTER, 20);
    //img1 = s.imageRoberts(img, AVERAGE_FILTER,20);
    //img2 = s.imageRoberts(img, MEDIAN_FILTER,20);
    //img3 = s.imageRoberts(img, GAUSSIAN_FILTER,20);
    //img4 = s.imageRoberts(img, BILATERAL_FILTER,20);
    //img5 = s.imageRoberts(img, SMALLWAVE_FILTER,20);
    // 彩色图像的图像增强测试
    //img0 = s.imageAverageFilter(img, 3, 3);
    //img1 = s.imageMedianFilter(img, 3, 3);
    //img2 = s.imageGaussianFilter(img, 3, 3);
    //img3 = s.imageBilateralFilter(img, 3, 3);
    //img4 = s.imageWaveletFilter(img, 3);
    //Prewitt边缘检测测试
    //img0 = s.imagePrewitt(img, NO_FILTER);
    //img1 = s.imagePrewitt(img, AVERAGE_FILTER);
    //img2 = s.imagePrewitt(img, MEDIAN_FILTER);
    //img3 = s.imagePrewitt(img, GAUSSIAN_FILTER);
    //img4 = s.imagePrewitt(img, BILATERAL_FILTER);
    //img5 = s.imagePrewitt(img, SMALLWAVE_FILTER);
    //Sobel边缘检测测试-核大小为3
    //img0 = s.imageSobel(img, NO_FILTER);
    //img1 = s.imageSobel(img, AVERAGE_FILTER);
    //img2 = s.imageSobel(img, MEDIAN_FILTER);
    //img3 = s.imageSobel(img, GAUSSIAN_FILTER);
    //img4 = s.imageSobel(img, BILATERAL_FILTER);
    //img5 = s.imageSobel(img, SMALLWAVE_FILTER);
    //Sobel边缘检测测试-核大小为5   
    //img0 = s.imageSobel(img, NO_FILTER,5);
    //img1 = s.imageSobel(img, AVERAGE_FILTER, 5);
    //img2 = s.imageSobel(img, MEDIAN_FILTER, 5);
    //img3 = s.imageSobel(img, GAUSSIAN_FILTER, 5);
    //img4 = s.imageSobel(img, BILATERAL_FILTER, 5);
    //img5 = s.imageSobel(img, SMALLWAVE_FILTER, 5);
    //Kirsch边缘检测测试
    //img0 = s.imageKirsch(img, NO_FILTER);
    //img1 = s.imageKirsch(img, AVERAGE_FILTER);
    //img2 = s.imageKirsch(img, MEDIAN_FILTER);
    //img3 = s.imageKirsch(img, GAUSSIAN_FILTER);
    //img4 = s.imageKirsch(img, BILATERAL_FILTER);
    //img5 = s.imageKirsch(img, SMALLWAVE_FILTER);
    //Rbinson边缘检测测试
    //img0 = s.imageRobinson(img, NO_FILTER);
    //img1 = s.imageRobinson(img, AVERAGE_FILTER);
    //img2 = s.imageRobinson(img, MEDIAN_FILTER);
    //img3 = s.imageRobinson(img, GAUSSIAN_FILTER);
    //img4 = s.imageRobinson(img, BILATERAL_FILTER);
    //img5 = s.imageRobinson(img, SMALLWAVE_FILTER);
    //Laplacian边缘检测测试
    //img0 = s.imageLaplacian(img, NO_FILTER);
    //img1 = s.imageLaplacian(img, AVERAGE_FILTER);
    //img2 = s.imageLaplacian(img, MEDIAN_FILTER);
    //img3 = s.imageLaplacian(img, GAUSSIAN_FILTER);
    //img4 = s.imageLaplacian(img, BILATERAL_FILTER);
    //img5 = s.imageLaplacian(img, SMALLWAVE_FILTER);
    //Canny边缘检测测试
    //img0 = s.imageCanny(img);
    //图像对比度增强的测试
    //img0 = s.imageContrastEnhance(img,255);
    //图像亮度增强测试
    //img0 = s.imageBrightness(img, 20);
    //直方图均衡化测试
    //img0 = s.imageHistogramEqualization(img);
    //指数增强测试
    //img0 = s.imageExponentialTransform(img);
    /*图像打马赛克*/
    //img0 = s.imageMasaic(img, 5);
    /*图像卷积*/
    //int kernel_size = 3;
    //int** laplacian_kernel = new int* [kernel_size];
    //for (int i = 0; i < kernel_size; ++i) {
    //    laplacian_kernel[i] = new int[kernel_size];
    //}
    //laplacian_kernel[0][0] = 1, laplacian_kernel[0][1] = -1, laplacian_kernel[0][2] = -1;
    //laplacian_kernel[1][0] = -1, laplacian_kernel[1][1] = 4, laplacian_kernel[1][2] = -1;
    //laplacian_kernel[2][0] = -1, laplacian_kernel[2][1] = -1, laplacian_kernel[2][2] = 1;
    //img0 = s.imageCovolution(img, kernel_size, laplacian_kernel);

    //for (int i = 0; i < kernel_size; ++i) {
    //    delete[] laplacian_kernel[i];
    //}
    //delete[] laplacian_kernel;
   
    /*图像拼接测试*/
    //img0 = s.imageSynthesis(imgx, imgy);
    /*图像分割测试*/
    //img0 = s.imageSegmentation(img);
    //图像傅里叶变换测试
    //img0 = s.imageFourierTransform(img);
    //imshow("img0",img0);
    /*数字识别进行检测*/
    //img0=s.imageDigitalIdentify(img);
    //imshow("test", img0);
    return a.exec();
}
