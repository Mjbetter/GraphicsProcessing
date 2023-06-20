#pragma
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QToolBar>
#include <QLabel>
#include <QPixmap>
#include <qmenu.h>
#include <qmenubar.h>
#include <QVBoxLayout>
#include <QSplitter>
#include <QStackedLayout>
#include <core/base.hpp>
#include <opencv2/opencv.hpp>
#include <QStandardItemModel>
#include <QTreeView>
#include "Algorithm.h"
#include <stack>
#include <QPainter>
#include "Algorithm.h"
#include <QTimer>
#include <QPlainTextEdit>

using namespace cv;
using namespace std;

class UI : public QMainWindow
{

public:

	//test
	void translateImage(int dx, int dy);


	//设置主窗口
	void initmainwin(QMainWindow* mainwin);
	//在主窗口设置一个菜单用于一些快捷键
	void createUpMenu(QMainWindow* mainwin);
	QMenuBar* menu;
	QAction* openimage;
	QAction* saveimage;
	QAction* revokeac;
	QAction* redoac;

	//设置中心窗口
	void createCenterWin(QMainWindow* mainwin);
	//在中心窗口设置左侧菜单
	QStandardItemModel* createLeftMenu(QWidget* leftwidget);
	//设置右击菜单
	void right_clickMenu(QWidget* centralWidget);
	/*
	-------------------------------------------右击菜单----------------------------------------------------------------
	*/
	QMenu* clickmenu;

	/*
	-----------------------------------openImage&saveImage-------------------------------------------------------------
	*/
	//openImage&saveImage
	QLabel* imageLabel = nullptr;
	QString imagePath;
	QString savePath;
	/*
	-------------------------------------mainwin下的一级窗口-----------------------------------------------------------
	*/
	//中心窗口
	QWidget* centralWidget;
	//中心窗口总布局--将会水平分为左右两个窗口，左侧窗口用于放置菜单
	QHBoxLayout* mainlayout;
	/*
	-------------------------------------mainwin下的二级窗口-----------------------------------------------------------
	----------------------------------centralWidget下的一级窗口--------------------------------------------------------
	*/
	//左侧窗口及其布局
	QWidget* leftWidget;
	QVBoxLayout* leftWidgetLayout;
	//右侧窗口及其布局
	QWidget* rightWidget;
	QVBoxLayout* rightWidgetLayout;
	/*
	-------------------------------------mainwin下的二级窗口-----------------------------------------------------------
	----------------------------------centralWidget下的一级窗口--------------------------------------------------------
	------------------------------------leftWidget下的菜单布局---------------------------------------------------------
	*/
	//创建菜单布局,垂直分布
	QVBoxLayout* menubarLayout;
	//创建树视图
	QTreeView* treeView;
	//递归设置菜单
	void setFontAndSizeRecursive(QStandardItem* item, const QFont& font, int fontSize);

	/*
-------------------------------------------左侧菜单----------------------------------------------------------------
*/

	// 创建标准项模型_相当于创建左侧的菜单容器
	QStandardItemModel* MenuModel;

	//创建一阶菜单项
	QStandardItem* fileOP;
	QStandardItem* ImaAdjust;
	QStandardItem* ImaDetail;
	QStandardItem* ImaEdge;
	QStandardItem* ImaPro;
	QStandardItem* ImaCom;
	QStandardItem* ImaSeg;
	QStandardItem* ImaNumRec;
	QStandardItem* ImaSketching;

	//二阶菜单
	
	//fileOP
	QStandardItem* openAction;
	QStandardItem* saveAction;
	QStandardItem* vImaInfoAction;
	QStandardItem* rset;
	//ImaAdjust
	QStandardItem* ImaPanAction;
	int xNum;
	int yNum;
	QStandardItem* ImaZoomAction;
	double zoomNum;
	QStandardItem* ImaRotAction;
	int rotataNum;
	QStandardItem* ImaMirrAction;
	int MirrNum;
	//ImaDetail
	QStandardItem* ImaGS;
	QStandardItem* ImaNoiPro;
	QStandardItem* ImaPasEdg;
	QStandardItem* ImaShrEdg;
	QStandardItem* ImaHisEdg;
	//ImaEdge
	QStandardItem* ImaFirRober;
	QStandardItem* ImaFirSobel;
	QStandardItem* ImaFirPrewi;
	QStandardItem* ImaFirKirsc;
	QStandardItem* ImaFirRobin;
	QStandardItem* ImaFirLapla;
	QStandardItem* ImaFirCanny;
	//ImaPro
	QStandardItem* ImaEnh;
	QStandardItem* ImaMos;
	int MosaicNum;
	QStandardItem* ImaConv;
	int** matrix;
	int KernelSize;
	QStandardItem* ImaFourAnal;

	//三阶菜单
	//ImaDetail
	//ImaDetail_ImaGS
	QStandardItem* ImaGray;
	QStandardItem* ImaBin;
	//ImaDetail_ImaNoiPro
	QStandardItem* ImaNoiRe;
	QStandardItem* ImaNoiAdd;
	//ImaPro
	//ImaPro_ImaEnh
	QStandardItem* ImaCE;
	int ContrastNum;
	QStandardItem* ImaBE;
	int BrightnessNum;
	QStandardItem* ImaHE;
	int Exponential1Num;
	int Exponential2Num;
	QStandardItem* ImaETE;

	//四阶菜单
	//ImaDetail_ImaNoiPro_ImaNoiRe
	int Con_KernelSize;
	QStandardItem* ImaMeanF;
	QStandardItem* ImaMediF;
	QStandardItem* ImaGausF;
	QStandardItem* ImaBiluF;
	QStandardItem* ImaWaveF;
	//ImaDetail_ImaNoiPro_ImaNoiAdd;
	QStandardItem* ImaGausN;
	QStandardItem* ImaSAPN;
	QStandardItem* ImaPoiN;

	/*
	-------------------------------------mainwin下的二级窗口-----------------------------------------------------------
	----------------------------------centralWidget下的一级窗口--------------------------------------------------------
	------------------------------------rightWidget下的图像与控件容器布局---------------------------------------------------------
	*/
	// 创建图像控件的布局管理器
	//QLabel* imageLabel = nullptr;
	QVBoxLayout* imageLayout;
	//控件容器布局管理
	QWidget* controlContainer;
	QHBoxLayout* controlLayout;
	//给slider的样貌变换一下
	void setSliderStyle(QSlider* slider);
	/*
	----------------------------------------mat与label的互相转换--------------------------------------------------------
	*/
	//将Mat图像转化成Label形式
	QPixmap convertMatToQPixmap(const cv::Mat& mat);
	//将Label图像转化成Mat形式
	cv::Mat convertQPixmapToMat(QPixmap pixmap);

private:
	bool isInputComplete = false;
public	slots:
	
	//根据索引寻找要实现的槽函数
	void handleMenuItemClicked(const QModelIndex& index);

	//删除控件变化区的存在控件
	void deleteChildWidgets(QWidget* parentWidget);

	//文件
	//打开图像文件
	void openImage();
	//保存图像文件
	void saveImage();
	//查看图像信息
	void showImageInfo();
	//文件操作重置
	void imageRest();

	//图像调整
	//平移
	void panImage();
	void pan_Image();
	//缩放
	void zoomImage();
	void zoom_Image();
	//旋转
	void rotataImage();
	void rotata_Image();
	//镜像
	void mirrorImage();
	void mirror_Image();

	//细节处理
	//变灰度
	//_灰度图
	void GrayImage();
	//_2值图
	void BinaryImage();
	//噪声处理
	//_去噪
	//__均值滤波
	void MeanF();
	void executeMeanF(int value);
	void mean_f();
	//__中值滤波
	void MedianF();
	void executeMedianF(int value);
	void median_f();
	//__高斯滤波
	void GaussianF();
	void executeGaussianF(int value);
	void gaussian_f();
	//__双边滤波
	void BilateralF();
	void executeBilateralF(int value);
	void bilateral_f();
	//__小波滤波
	void WaveletF();
	void executeWaveletF(int value);
	void wavelet_f();
	//_加噪
	//__高斯噪声
	void GaussianN();
	//__椒盐噪声
	void SaltAndPepperN();
	//__泊松噪声
	void PoissonN();
	//__钝化边缘
	void BluntE();
	//__锐化边缘
	void SharpE();
	//__图像直方图
	void HistogramE();

	//边缘提取
	//_Roberts算子
	void RobertsE();
	//_Sobel算子
	void SobelE();
	//_Prewitt算子
	void PrewittE();
	//_Kirsch算子
	void KirschE();
	//_Robinsom算子
	void RobinsomE();
	//_Laplacian算子
	void LaplacianE();
	//_Canny算子
	void CannyE();
	//图像处理
	//图像增强
	//_对比度增强
	void ContrastE();
	void contrast_e();
	//_亮度增强
	void BrightnessE();
	void brightness_e();
	//_直方图均衡化
	void HistogramEqualization();
	//_指数变化增强
	void ExponentialTransformationEnhancement();
	void Exponential_transformation_enhancement();
	//加马赛克
	void Mosaic();
	void mosaic();
	//图像卷积
	void ConvolutionImage();
	void convolution();
	//傅里叶变换
	void FourierTransform();

	//图像合成
	void ImageSynthesis();
	//图像分割
	void ImageSegmentation();
	//图像数字识别
	void ImageDigitRecognition();
	//图像素描化
	void ImageSketching();
	//右击菜单_撤销与反撤销
	//撤销
	void Revoke_operation();
	//反撤销
	void Redo_Operatio();
	//替换画布上图案
	void Replace_Picture(Mat img);

	//撤销，反撤销的栈
	stack<QPixmap> revoke;
	stack<QPixmap> redo;

	//进入一个功能时，画布上面的图片
	QPixmap nowPixmap;
};
