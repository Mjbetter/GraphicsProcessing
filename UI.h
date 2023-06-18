
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
#include "Algorithm.h"


class UI : public QWidget
{

public:


	//设置主窗口
	void initmainwin(QMainWindow* mainwin);

	//设置菜单
	void createMenu(QMainWindow* mainwin);
	//设置工具栏
	//void createToolbar(QMainWindow* mainwin);
	//设置中心窗口
	void createCenterWin(QMainWindow* mainwin);
	//设置右击菜单
	void right_clickMenu(QMainWindow* mainwin);
	QMenu* clickmenu;
	//上方菜单
	QMenuBar* menu;
	//一阶菜单
	QMenu* fileOP;
	QMenu* ImaAdjust;
	QMenu* ImaDetail;
	QMenu* ImaEdge;
	QMenu* ImaPro;
	QAction* ImaCom;
	QAction* ImaSeg;
	QAction* ImaNumRec;
	QMenu* ElseFunc;

	//二阶菜单
	//fileOP
	QAction* openAction;
	QAction* saveAction;
	QAction* vImaInfoAction;
	//ImaAdjust
	QAction* ImaPanAction;
	QAction* ImaZoomAction;
	QAction* ImaRotAction;
	QAction* ImaMirrAction;
	//ImaDetail
	QMenu* ImaGS;
	QMenu* ImaNoiPro;
	QAction* ImaPasEdg;
	QAction* ImaShrEdg;
	//ImaEdge
	QAction* ImaFirRober;
	QAction* ImaFirSobel;
	QAction* ImaFirPrewi;
	QAction* ImaFirKirsc;
	QAction* ImaFirRobin;
	//ImaPro
	QMenu* ImaEnh;
	QAction* ImaMos;
	QAction* ImaConv;
	QAction* ImaFourAnal;

	//三阶菜单
	//ImaDetail
	//ImaDetail_ImaGS
	QAction* ImaGray;
	QAction* ImaBin;
	//ImaDetail_ImaNoiPro
	QMenu* ImaNoiRe;
	QMenu* ImaNoiAdd;
	//ImaPro
	//ImaPro_ImaEnh
	QAction* ImaCE;
	QAction* ImaBE;
	QAction* ImaHE;
	QAction* ImaETE;

	//四阶菜单
	//ImaDetail_ImaNoiPro_ImaNoiRe
	QAction* ImaMeanF;
	QAction* ImaMediF;
	QAction* ImaGausF;
	QAction* ImaBiluF;
	QAction* ImaWaveF;
	//ImaDetail_ImaNoiPro_ImaNoiAdd;
	QAction* ImaGausN;
	QAction* ImaSAPN;
	QAction* ImaPoiN;



	//openImage&saveImage
	QLabel* imageLabel = nullptr;
	QString imagePath;
	QString savePath;


	QWidget* centralWidget;
	//QVBoxLayout* mainlayout;
	QStackedLayout* stackedLayout;

	// 创建图像控件的布局管理器
	QVBoxLayout* imageLayout;
	//图像与控件呈垂直分布
	QVBoxLayout* layout;
	//切换控件的页面
	QWidget* controlContainer;
	QHBoxLayout* controlLayout;
	//将Mat图像转化成Label形式
	QLabel* convertMatToQLabel(const cv::Mat& mat);
	//将Label图像转化成Mat形式
	cv::Mat convertQLabelToMat(const QLabel* imagelabel);
private:

public	slots:

	//删除控件变化区的存在控件
	void deleteChildWidgets(QWidget* parentWidget);

	//文件
	//打开图像文件
	void openImage();
	//保存图像文件
	void saveImage();
	//查看图像信息
	void showImageInfo();

	//图像调整
	//平移
	void panImage();
	//缩放
	void zoomImage();
	//旋转
	void rotataImage();
	//镜像
	void mirrorImage();

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
	//__中值滤波
	void MedianF();
	//__高斯滤波
	void GaussianF();
	//__双边滤波
	void BilateralF();
	//__小波滤波
	void WaveletF();
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

	//边缘提取
	//_Roberts算子
	void RobertsE();
	//_Sobel算子
	void SobelE();
	//_Prewitt算子
	void PrewittE();
	//_Kirsch算子
	void KirschE();
	//_Robinson算子
	void RobinsonE();
	//Laplacian算子
	void LaplacianE();
	//Canny算子
	void CannyE();

	//图像处理
	//图像增强
	//_对比度增强
	void ContrastE();
	//_亮度增强
	void BrightnessE();
	//_直方图均衡化
	void HistogramEqualization();
	//_指数变化增强
	void ExponentialTransformationEnhancement();
	//加马赛克
	void Mosaic();
	//图像卷积
	void ConvolutionImage();
	//傅里叶变换
	void FourierTransform();

	//图像合成
	void ImageSynthesis();
	//图像分割
	void ImageSegmentation();
	//图像数字识别
	void ImageDigitRecognition();
	//右击菜单_撤销与反撤销
	//撤销
	//void Revoke_operation();
	//反撤销
	//void Redo_Operatio();

};