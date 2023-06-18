#pragma once
#include "UI.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QMenuBar>
#include <qmenu.h>
#include <qmenubar.h>
#include <QVBoxLayout>
#include <ui_GraphicsProcessing.h>
#include <QPushButton>
#include <QDateTime>
#include <QScrollBar>
#include <QStackedWidget>
#include <QStackedLayout>
/*
函数作用:获得全屏的窗口，可放大缩小
函数参数:1、mainwin：指向窗口的指针
*/
void UI::initmainwin(QMainWindow* mainwin)
{
    mainwin->setWindowState(Qt::WindowMaximized);
    createMenu(mainwin);
    createCenterWin(mainwin);
    right_clickMenu(mainwin);
}

/*
--------------------------------------布局处理_删除控件区域的控件------------------------------------
*/

/*
函数作用:将放置控件区域清空
函数参数:1、parentWidget：被清空的子窗口
         2、Childlayout：子窗口布局
*/
void UI::deleteChildWidgets(QWidget* parentWidget)
{
    if (parentWidget)
    {
        QLayout* Childlayout = parentWidget->layout();
        if (Childlayout)
        {
            QLayoutItem* item;
            while ((item = Childlayout->takeAt(0)))
            {
                if (QWidget* childwidget = item->widget())
                {
                    delete childwidget;
                }
                else if (QLayout* childLayout = item->layout())
                {
                    deleteChildWidgets(childLayout->widget());
                }
                delete item;
            }
        }
    }
}
/*
--------------------------------------Mat与QLabel的相互转换------------------------------------------
*/
/*
函数作用:将 OpenCV 的 BGR 图像转换为 Label标签图像
函数参数:1、mat：OpenCV 的 BGR 图像
         2、conImalabel：用于放置转化为标签的图像
*/
QLabel* UI::convertMatToQLabel(const cv::Mat& mat)
{
    QLabel* conImalabel = new QLabel();

    if (mat.empty()) {
        conImalabel->setText("Empty Image");
        return conImalabel;
    }
    // 将 OpenCV 的 BGR 图像转换为 RGB 图像
    cv::Mat rgbImage;
    cv::cvtColor(mat, rgbImage, cv::COLOR_BGR2RGB);

    // 创建 Qt 图像对象
    QImage conimage(rgbImage.data, rgbImage.cols, rgbImage.rows, QImage::Format_RGB888);

    // 缩放图像以适应 QLabel
    QSize imageSize = conImalabel->size();
    QImage scaledImage = conimage.scaled(imageSize, Qt::KeepAspectRatio);
    return conImalabel;
}
/*
函数作用:将 Label标签图像转换为 OpenCV 的 BRG 图像
函数参数:1、imagelabel：被改的Label标签图像
         2、mat：转换的 OpenCV 的 BRG 图像
         3、pixmap、image：中间过程
*/
cv::Mat UI::convertQLabelToMat(const QLabel* imagelabel)
{
    cv::Mat mat;

    // 获取 QLabel 的图像
    QPixmap pixmap = *imagelabel->pixmap();
    QImage image = pixmap.toImage();

    // 将 QImage 转换为 OpenCV 的 Mat
    if (!image.isNull()) {
        QImage rgbImage = image.convertToFormat(QImage::Format_RGB888);
        mat = cv::Mat(rgbImage.height(), rgbImage.width(), CV_8UC3, rgbImage.bits(), rgbImage.bytesPerLine()).clone();
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    }

    return mat;
}

/*
------------------------------------------------文件---------------------------------------------------------------
*/

/*
函数作用:打开图像文件
函数参数:1、imagePath：文件路径
         2、image：一个QPixmap类的对象
         3、imageLabel：将image设置成图像标签的像素图
*/
void UI::openImage()
{
    /*
        它打开一个文件对话框，允许用户选择一个带有扩展名".png"、".jpg"或".bmp"的图像文件。
        选择的文件路径存储在变量imagePath中。不过这个imagePath是一个私有变量。
    */
    imagePath = QFileDialog::getOpenFileName(this, "Open Image", QString(), "Image Files (*.png *.jpg *.bmp *.webp)");
    if (!imagePath.isEmpty()) {
        //它使用选定的文件路径创建一个名为image的QImage对象。
        QImage image(imagePath);
        /*
            image.isNull检查QPixmap对象是否为空。
            如果QPixmap对象为空，它使用QMessageBox显示一个错误消息。
        */
        if (image.isNull()) {
            QMessageBox::information(this, "Error", "Failed to open image.");
        }
        /*
            如果image不为空，将image设置为名为imageLabel的图像标签的像素图。
        */
        else {
            image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            QPixmap pixmap = QPixmap::fromImage(image);
            imageLabel->setPixmap(pixmap);
            //imageLabel->adjustSize();        //调整图像标签的大小以适应加载的图像的大小。
        }
    }
}

/*
函数作用:保存图像文件
函数参数:1、imagePath：文件路径
         2、image：一个QPixmap类的对象
         3、imageLabel：将image设置成图像标签的像素图
*/
void UI::saveImage()
{
    savePath = QFileDialog::getSaveFileName(this, "Save Image", QString(), "Image Files (*.png *.jpg *.bmp)");
    if (!savePath.isEmpty()) {
        //const关键字是为了指示pixmap指针所指向的QPixmap对象是只读的，即不允许修改该对象。
        const QPixmap* pixmap = imageLabel->pixmap(); // 获取图像标签中的像素图指针
        if (pixmap) {
            QPixmap image = pixmap->copy(); // 使用copy函数创建新的QPixmap对象
            bool saved = image.save(savePath);
            if (saved) {
                QMessageBox::information(this, "Success", "Image saved successfully.");
            }
            else {
                QMessageBox::information(this, "Error", "Failed to save image.");
            }
        }
        else {
            QMessageBox::information(this, "Error", "No image to save.");
        }
    }
}

/*
函数作用:查看图像信息
函数参数:1、imagePath：在openImage中获取的图像路径
         2、fileInfo：图像文件信息类
         3、fileName：图像文件名字
         4、fileSize：图像文件大小
         5、fileCreated：图像文件创建时间
         6、fileModified：图像文件最后一次修改的时间
*/
void UI::showImageInfo()
{
    if (!imagePath.isEmpty())
    {
        QFileInfo fileInfo(imagePath);
        QString fileName = fileInfo.fileName();
        QString fileSize = QString::number(fileInfo.size());
        QString fileCreated = fileInfo.created().toString();
        QString fileModified = fileInfo.lastModified().toString();

        QMessageBox::information(this, "Image Information", "File Name: " + fileName +
            "\nFile Size: " + fileSize +
            "\nCreated: " + fileCreated +
            "\nModified: " + fileModified);
    }
    else
    {
        QMessageBox::information(this, "Error", "No information of image to show.");
    }
}

/*
----------------------------------------------------图像调整-------------------------------------------------------
*/
/*
函数作用：通过窗口获取xy轴的数值来进行图像平移
函数参数：
*/
void UI::panImage()
{
    int x;
    int y;
    QSlider* sliderX = new QSlider(Qt::Horizontal, this); // 为X创建水平滚动条
    QSlider* sliderY = new QSlider(Qt::Horizontal, this); // 为Y创建水平滚动条
    //X
    sliderX->setMinimum(0); // 设置最小值
    sliderX->setMaximum(100); // 设置最大值
    sliderX->setValue(50); // 设置初始值
    sliderX->setSingleStep(1); // 设置步长
    //Y
    sliderY->setMinimum(0); // 设置最小值
    sliderY->setMaximum(100); // 设置最大值
    sliderY->setValue(50); // 设置初始值
    sliderY->setSingleStep(1); // 设置步长

    QLabel* labelx = new QLabel(QString::number(sliderX->value()), this);
    QLabel* labely = new QLabel(QString::number(sliderY->value()), this);

    x = sliderX->value();
    y = sliderY->value();

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(labelx);
    controlLayout->addWidget(sliderX);
    controlLayout->addWidget(sliderY);
    controlLayout->addWidget(labely);

    connect(sliderX, &QSlider::valueChanged, this, [labelx](int value) {
        labelx->setText(QString::number(value));
        });
    connect(sliderY, &QSlider::valueChanged, this, [labely](int value) {
        labely->setText(QString::number(value));
        });

}
/*
函数作用：通过窗口获取放大或缩小的数值来进行图像缩放
函数参数：
*/
void UI::zoomImage()
{
    int zoomNum;
    QSlider* sliderZoom = new QSlider(Qt::Horizontal, this); // 为zoomNum创建水平滚动条
    //zoomNum
    sliderZoom->setMinimum(-100); // 设置最小值
    sliderZoom->setMaximum(100); // 设置最大值
    sliderZoom->setValue(0); // 设置初始值
    sliderZoom->setSingleStep(1); // 设置步长
    QLabel* labelZoom = new QLabel(QString::number(sliderZoom->value()), this);
    zoomNum = sliderZoom->value();

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderZoom);
    controlLayout->addWidget(labelZoom);
    connect(sliderZoom, &QSlider::valueChanged, this, [labelZoom](int value) {
        labelZoom->setText(QString::number(value));
        });
}
/*
函数作用：通过窗口获取旋转的数值来进行图像旋转
函数参数：
*/
void UI::rotataImage()
{
    int rotataNum;
    QSlider* sliderRotata = new QSlider(Qt::Horizontal, this); // 为zoomNum创建水平滚动条
    //zoomNum
    sliderRotata->setMinimum(-180); // 设置最小值
    sliderRotata->setMaximum(180); // 设置最大值
    sliderRotata->setValue(0); // 设置初始值
    sliderRotata->setSingleStep(1); // 设置步长
    QLabel* labelRotata = new QLabel(QString::number(sliderRotata->value()), this);
    rotataNum = sliderRotata->value();

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderRotata);
    controlLayout->addWidget(labelRotata);
    connect(sliderRotata, &QSlider::valueChanged, this, [labelRotata](int value) {
        labelRotata->setText(QString::number(value));
        });
}
/*
函数作用：直接调用函数进行图像镜像变换
函数参数：
*/
void UI::mirrorImage()
{

}

/*
----------------------------------------------------细节处理-------------------------------------------------------
*/
//变灰度
//_灰度图
/*
函数作用：
函数参数：
*/
void UI::GrayImage()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//_2值图
/*
函数作用：
函数参数：
*/
void UI::BinaryImage()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//去噪
//均值滤波
/*
函数作用：
函数参数：
*/
void UI::MeanF()
{
    /*将控件上的图片转化为img*/
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    /*进行均值滤波*/
    Mat newImage = method.imageDenoising(img, 5, AVERAGE_FILTER);
    /*将图片转化为RGB格式*/
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    /*Mat类型转化为QImage格式*/
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);

    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

    QPixmap pixmap = QPixmap::fromImage(image);
        
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    imageLabel->setPixmap(pixmap);
}
//中值滤波
/*
函数作用：
函数参数：
*/
void UI::MedianF()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageDenoising(img, 9, MEDIAN_FILTER);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data,newImage.cols,newImage.rows,newImage.step,QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//高斯滤波
/*
函数作用：
函数参数：
*/
void UI::GaussianF()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageDenoising(img, 9, GAUSSIAN_FILTER);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//双边滤波
/*
函数作用：
函数参数：
*/
void UI::BilateralF()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageDenoising(img, 9, BILATERAL_FILTER);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//小波滤波
/*
函数作用：
函数参数：
*/
void UI::WaveletF()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageDenoising(img, 9, SMALLWAVE_FILTER);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//加噪
//高斯噪声
/*
函数作用：
函数参数：
*/
void UI::GaussianN()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//椒盐噪声
/*
函数作用：
函数参数：
*/
void UI::SaltAndPepperN()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//泊松噪声
/*
函数作用：
函数参数：
*/
void UI::PoissonN()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//钝化边缘
/*
函数作用：
函数参数：
*/
void UI::BluntE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//锐化边缘
/*
函数作用：
函数参数：
*/
void UI::SharpE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}


/*
----------------------------------------------------边缘提取------------------------------------------------
*/
//Roberts算子
/*
函数作用：
函数参数：
*/
void UI::RobertsE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, ROBERTS);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//Sobel算子
/*
函数作用：
函数参数：
*/
void UI::SobelE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, SOBEL);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//Prewitt算子
/*
函数作用：
函数参数：
*/
void UI::PrewittE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, PREWITT);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//Kirsch算子
/*
函数作用：
函数参数：
*/
void UI::KirschE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, KIRSCH);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}

//Robinsom算子
/*
函数作用：
函数参数：
*/
void UI::RobinsonE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, ROBINSON);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}

void UI::LaplacianE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, LAPLACIAN);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
void UI::CannyE()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEdgeDetection(img, CANNY);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}

/*
----------------------------------------------------图像处理------------------------------------------------
*/
//图像增强
//_对比度增强
/*
函数作用：
函数参数：
*/
void UI::ContrastE()
{
    int ContrastNum;
    QSlider* sliderContrast = new QSlider(Qt::Horizontal, this); // 为ContrastNum创建水平滚动条
    //ContrastNum
    sliderContrast->setMinimum(0); // 设置最小值
    sliderContrast->setMaximum(100); // 设置最大值
    sliderContrast->setValue(0); // 设置初始值
    sliderContrast->setSingleStep(1); // 设置步长
    QLabel* labelRotata = new QLabel(QString::number(sliderContrast->value()), this);
    ContrastNum = sliderContrast->value();

    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEnhance(img, CONTRAST_ENHANCE,sliderContrast->value());
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
    controlLayout->addWidget(sliderContrast);
    controlLayout->addWidget(labelRotata);
    connect(sliderContrast, &QSlider::valueChanged, this, [labelRotata](int value) {
        labelRotata->setText(QString::number(value));
        });
}
//_亮度增强
/*
函数作用：
函数参数：
*/
void UI::BrightnessE()
{
    int BrightnessNum;
    QSlider* sliderBrightness = new QSlider(Qt::Horizontal, this); // 为BrightnessNum创建水平滚动条
    //ContrastNum
    sliderBrightness->setMinimum(-100); // 设置最小值
    sliderBrightness->setMaximum(100); // 设置最大值
    sliderBrightness->setValue(0); // 设置初始值
    sliderBrightness->setSingleStep(1); // 设置步长
    QLabel* labelBrightness = new QLabel(QString::number(sliderBrightness->value()), this);
    BrightnessNum = sliderBrightness->value();

    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEnhance(img, BRIGHTNESS, sliderBrightness->value());
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
    controlLayout->addWidget(sliderBrightness);
    controlLayout->addWidget(labelBrightness);
    connect(sliderBrightness, &QSlider::valueChanged, this, [labelBrightness](int value) {
        labelBrightness->setText(QString::number(value));
        });
}
//_直方图均衡化
/*
函数作用：
函数参数：
*/
void UI::HistogramEqualization()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageEnhance(img, HISTOGRAME_QUALIZATION);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}
//_指数变化增强
void UI::ExponentialTransformationEnhancement()
{

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//加马赛克
/*
函数作用：
函数参数：
*/
void UI::Mosaic()
{
    int MosaicNum;

    QSlider* sliderMosaic = new QSlider(Qt::Horizontal, this); // 为MosaicNum创建水平滚动条
    //ContrastNum
    sliderMosaic->setMinimum(0); // 设置最小值
    sliderMosaic->setMaximum(50); // 设置最大值
    sliderMosaic->setValue(0); // 设置初始值
    sliderMosaic->setSingleStep(1); // 设置步长
    QLabel* labelMosaic = new QLabel(QString::number(sliderMosaic->value()), this);
    MosaicNum = sliderMosaic->value();

    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageMasaic(img,sliderMosaic->value());
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
    controlLayout->addWidget(sliderMosaic);
    controlLayout->addWidget(labelMosaic);
    connect(sliderMosaic, &QSlider::valueChanged, this, [labelMosaic](int value) {
        labelMosaic->setText(QString::number(value));
        });
}
//图像卷积
/*
函数作用：
函数参数：
*/
void UI::ConvolutionImage()
{
    int KernelNum;
    int ChannelNum;

    QSlider* sliderKernel = new QSlider(Qt::Horizontal, this); // 为KernelNum创建水平滚动条
    QSlider* sliderChannel = new QSlider(Qt::Horizontal, this); // 为ChannelNum创建水平滚动条
    //KernelNum
    sliderKernel->setMinimum(0); // 设置最小值
    sliderKernel->setMaximum(10); // 设置最大值
    sliderKernel->setValue(0); // 设置初始值
    sliderKernel->setSingleStep(1); // 设置步长
    QLabel* labelKernel = new QLabel(QString::number(sliderKernel->value()), this);
    KernelNum = sliderKernel->value();
    //ChannelNum
    sliderChannel->setMinimum(0); // 设置最小值
    sliderChannel->setMaximum(10); // 设置最大值
    sliderChannel->setValue(0); // 设置初始值
    sliderChannel->setSingleStep(1); // 设置步长
    QLabel* labelChannel = new QLabel(QString::number(sliderChannel->value()), this);
    KernelNum = sliderChannel->value();

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(labelKernel);
    controlLayout->addWidget(sliderKernel);
    controlLayout->addWidget(sliderChannel);
    controlLayout->addWidget(labelChannel);
    connect(sliderKernel, &QSlider::valueChanged, this, [labelKernel](int value) {
        labelKernel->setText(QString::number(value));
        });
    connect(sliderChannel, &QSlider::valueChanged, this, [labelChannel](int value) {
        labelChannel->setText(QString::number(value));
        });
}
//傅里叶变换
/*
函数作用：
函数参数：
*/
void UI::FourierTransform()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageFourierTransform(img);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}

/*
-------------------------------------------图像合成---------------------------------------------------------------
*/
/*
函数作用：
函数参数：
*/
void UI::ImageSynthesis()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    QString OtherimagePath = QFileDialog::getOpenFileName(this, "Select Image", QString(), "Image Files (*.png *.jpg *.bmp)");
    if (!OtherimagePath.isEmpty()) {
        //它使用选定的文件路径创建一个名为image的QImage对象。
        QImage Otherimage(OtherimagePath);
        /*
            image.isNull检查QPixmap对象是否为空。
            如果QPixmap对象为空，它使用QMessageBox显示一个错误消息。
        */
        if (Otherimage.isNull()) {
            QMessageBox::information(this, "Error", "Failed to open image.");
        }
        /*
            如果image不为空，将image设置为名为imageLabel的图像标签的像素图。
        */
        else {
            Otherimage = Otherimage.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            QImage rgbaOtherImage = Otherimage.convertToFormat(QImage::Format_RGBA8888);
            cv::Mat Othermat(rgbaOtherImage.height(), rgbaOtherImage.width(), CV_8UC4, rgbaOtherImage.bits(), rgbaOtherImage.bytesPerLine());
            cv::cvtColor(Othermat, Othermat, cv::COLOR_RGB2BGR);
            Mat img = convertQLabelToMat(imageLabel);
            ImageAlgorithm method;
            Mat newImage = method.imageSynthesis(Othermat,img);
            cvtColor(newImage, newImage, COLOR_BGR2RGB);
            QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
            image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            QPixmap pixmap = QPixmap::fromImage(image);
            //删除空间变换区域原有控件
            imageLabel->setPixmap(pixmap);
        }
    }


}
/*
-------------------------------------------图像分割---------------------------------------------------------------
*/
/*
函数作用：
函数参数：
*/
void UI::ImageSegmentation()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageSegmentation(img);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}

/*
-------------------------------------------图像数字识别---------------------------------------------------------------
*/
/*
函数作用：
函数参数：
*/
void UI::ImageDigitRecognition()
{
    Mat img = convertQLabelToMat(imageLabel);
    ImageAlgorithm method;
    Mat newImage = method.imageDigitalIdentify(img);
    cvtColor(newImage, newImage, COLOR_BGR2RGB);
    QImage image(newImage.data, newImage.cols, newImage.rows, newImage.step, QImage::Format_RGB888);
    image = image.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap pixmap = QPixmap::fromImage(image);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    imageLabel->setPixmap(pixmap);
}

/*
-----------------------------------------右击菜单_撤销与反撤销--------------------------------------------------
*/
//撤销
void Revoke_operation()
{

}
//反撤销
void Redo_Operatio()
{

}

/*
-------------------------------------------页面布局---------------------------------------------------------------
*/
/*
函数作用:设置左侧菜单栏
函数参数:1、fileMenu:
         2、openAction：
*/
void UI::createMenu(QMainWindow* mainwin)
{
    menu = new QMenuBar(mainwin);
    mainwin->setMenuBar(menu);

    //一级菜单

    fileOP = new QMenu(" 文件 ", menu);
    menu->addMenu(fileOP);

    ImaAdjust = new QMenu(" 图像调整 ", menu);
    menu->addMenu(ImaAdjust);

    ImaDetail = new QMenu(" 细节处理 ", menu);
    menu->addMenu(ImaDetail);

    ImaEdge = new QMenu(" 边缘提取 ", menu);
    menu->addMenu(ImaEdge);

    ImaPro = new QMenu(" 图像处理 ", menu);
    menu->addMenu(ImaPro);

    ImaCom = new QAction(" 图像合成 ", this);
    menu->addAction(ImaCom);
    connect(ImaCom, &QAction::triggered, this, &UI::ImageSynthesis);

    ImaSeg = new QAction(" 图像分割 ", this);
    menu->addAction(ImaSeg);
    connect(ImaSeg, &QAction::triggered, this, &UI::ImageSegmentation);

    ImaNumRec = new QAction(" 图像数字识别 ", this);
    menu->addAction(ImaNumRec);
    connect(ImaNumRec, &QAction::triggered, this, &UI::ImageDigitRecognition);

    ElseFunc = new QMenu(" 其他 ", menu);
    menu->addMenu(ElseFunc);

    //二级菜单

    //文件的二级菜单

    openAction = new QAction(" 打开文件 ", this);
    fileOP->addAction(openAction);
    connect(openAction, &QAction::triggered, this, &UI::openImage);

    saveAction = new QAction(" 保存当前文件 ", this);
    fileOP->addAction(saveAction);
    connect(saveAction, &QAction::triggered, this, &UI::saveImage);

    vImaInfoAction = new QAction(" 查看当前文件信息 ", this);
    fileOP->addAction(vImaInfoAction);
    connect(vImaInfoAction, &QAction::triggered, this, &UI::showImageInfo);

    //图像调整的二级菜单

    ImaPanAction = new QAction(" 平移 ", this);
    ImaAdjust->addAction(ImaPanAction);
    connect(ImaPanAction, &QAction::triggered, this, &UI::panImage);

    ImaZoomAction = new QAction(" 缩放 ", this);
    ImaAdjust->addAction(ImaZoomAction);
    connect(ImaZoomAction, &QAction::triggered, this, &UI::zoomImage);

    ImaRotAction = new QAction(" 旋转 ", this);
    ImaAdjust->addAction(ImaRotAction);
    connect(ImaRotAction, &QAction::triggered, this, &UI::rotataImage);

    ImaMirrAction = new QAction(" 镜像 ", this);
    ImaAdjust->addAction(ImaMirrAction);
    connect(ImaMirrAction, &QAction::triggered, this, &UI::mirrorImage);

    //细节处理的二级菜单
    ImaGS = new QMenu(" 变灰度 ", ImaDetail);
    ImaDetail->addMenu(ImaGS);

    ImaNoiPro = new QMenu(" 噪声处理 ", ImaDetail);
    ImaDetail->addMenu(ImaNoiPro);

    ImaPasEdg = new QAction(" 边缘钝化 ", this);
    ImaDetail->addAction(ImaPasEdg);
    connect(ImaPasEdg, &QAction::triggered, this, &UI::BluntE);

    ImaShrEdg = new QAction(" 边缘锐化 ", this);
    ImaDetail->addAction(ImaShrEdg);
    connect(ImaShrEdg, &QAction::triggered, this, &UI::SharpE);

    //边缘提取的二级菜单
    ImaFirRober = new QAction(" Roberts算子 ", this);
    ImaEdge->addAction(ImaFirRober);
    connect(ImaFirRober, &QAction::triggered, this, &UI::RobertsE);

    ImaFirSobel = new QAction(" Sobel算子 ", this);
    ImaEdge->addAction(ImaFirSobel);
    connect(ImaFirSobel, &QAction::triggered, this, &UI::SobelE);

    ImaFirPrewi = new QAction(" Prewitt算子 ", this);
    ImaEdge->addAction(ImaFirPrewi);
    connect(ImaFirPrewi, &QAction::triggered, this, &UI::PrewittE);

    ImaFirKirsc = new QAction(" Kirsch算子 ", this);
    ImaEdge->addAction(ImaFirKirsc);
    connect(ImaFirKirsc, &QAction::triggered, this, &UI::KirschE);

    ImaFirRobin = new QAction(" Robinsom算子 ", this);
    ImaEdge->addAction(ImaFirRobin);
    connect(ImaFirRobin, &QAction::triggered, this, &UI::RobinsonE);

    ImaFirRobin = new QAction(" Laplacian算子", this);
    ImaEdge->addAction(ImaFirRobin);
    connect(ImaFirRobin, &QAction::triggered, this, &UI::LaplacianE);

    ImaFirRobin = new QAction(" Canny算子 ", this);
    ImaEdge->addAction(ImaFirRobin);
    connect(ImaFirRobin, &QAction::triggered, this, &UI::CannyE);

    //图像处理的二级菜单
    ImaEnh = new QMenu(" 图像增强 ", ImaPro);
    ImaPro->addMenu(ImaEnh);
    ImaMos = new QAction(" 马赛克 ", this);
    ImaPro->addAction(ImaMos);
    connect(ImaMos, &QAction::triggered, this, &UI::Mosaic);
    ImaConv = new QAction(" 图像卷积 ", this);
    ImaPro->addAction(ImaConv);
    connect(ImaConv, &QAction::triggered, this, &UI::ConvolutionImage);
    ImaFourAnal = new QAction(" 傅里叶变换 ", this);
    ImaPro->addAction(ImaFourAnal);
    connect(ImaFourAnal, &QAction::triggered, this, &UI::FourierTransform);

    //三级菜单

    //细节处理的三级菜单
    //ImaDetail_ImaGS
    ImaGray = new QAction(" 灰度图 ", this);
    ImaGS->addAction(ImaGray);
    connect(ImaGray, &QAction::triggered, this, &UI::GrayImage);
    ImaBin = new QAction(" 2值图 ", this);
    ImaGS->addAction(ImaBin);
    connect(ImaBin, &QAction::triggered, this, &UI::BinaryImage);
    //ImaDetail_ImaNoiPro
    ImaNoiRe = new QMenu(" 去噪 ", this);
    ImaNoiPro->addMenu(ImaNoiRe);
    ImaNoiAdd = new QMenu(" 加噪 ", this);
    ImaNoiPro->addMenu(ImaNoiAdd);
    //图像处理的三级菜单
    //ImaPro_ImaEnh
    ImaCE = new QAction(" 对比度增强 ", this);
    ImaEnh->addAction(ImaCE);
    connect(ImaCE, &QAction::triggered, this, &UI::ContrastE);
    ImaBE = new QAction(" 亮度增强 ", this);
    ImaEnh->addAction(ImaBE);
    connect(ImaBE, &QAction::triggered, this, &UI::BrightnessE);
    ImaHE = new QAction(" 直方图均衡化 ", this);
    ImaEnh->addAction(ImaHE);
    connect(ImaHE, &QAction::triggered, this, &UI::HistogramEqualization);
    ImaETE = new QAction(" 指数变换增强 ", this);
    ImaEnh->addAction(ImaETE);
    connect(ImaETE, &QAction::triggered, this, &UI::ExponentialTransformationEnhancement);

    //四级菜单
    // 细节处理的四级菜单
    //ImaDetail_ImaNoiPro_ImaNoiRe
    ImaMeanF = new QAction(" 均值滤波 ", this);
    ImaNoiRe->addAction(ImaMeanF);
    connect(ImaMeanF, &QAction::triggered, this, &UI::MeanF);
    ImaMediF = new QAction(" 中值滤波 ", this);
    ImaNoiRe->addAction(ImaMediF);
    connect(ImaMediF, &QAction::triggered, this, &UI::MedianF);
    ImaGausF = new QAction(" 高斯滤波 ", this);
    ImaNoiRe->addAction(ImaGausF);
    connect(ImaGausF, &QAction::triggered, this, &UI::GaussianF);
    ImaBiluF = new QAction(" 双边滤波 ", this);
    ImaNoiRe->addAction(ImaBiluF);
    connect(ImaBiluF, &QAction::triggered, this, &UI::BilateralF);
    ImaWaveF = new QAction(" 小波滤波 ", this);
    ImaNoiRe->addAction(ImaWaveF);
    connect(ImaWaveF, &QAction::triggered, this, &UI::WaveletF);
    //ImaDetail_ImaNoiPro_ImaNoiAdd;
    ImaGausN = new QAction(" 高斯噪声 ", this);
    ImaNoiAdd->addAction(ImaGausN);
    connect(ImaGausN, &QAction::triggered, this, &UI::GaussianN);
    ImaSAPN = new QAction(" 椒盐噪声 ", this);
    ImaNoiAdd->addAction(ImaSAPN);
    connect(ImaSAPN, &QAction::triggered, this, &UI::SaltAndPepperN);
    ImaPoiN = new QAction(" 泊松噪声 ", this);
    ImaNoiAdd->addAction(ImaPoiN);
    connect(ImaPoiN, &QAction::triggered, this, &UI::PoissonN);
}
/*
函数作用:设置中心窗口
函数参数:1、centralWidget
         2、layout
         3、imageLabel
*/
void UI::createCenterWin(QMainWindow* mainwin)
{
    centralWidget = new QWidget(mainwin);
    mainwin->setCentralWidget(centralWidget);
    // 创建图像控件的布局管理器
    imageLayout = new QVBoxLayout();

    //// 获取主窗口大小
    //QSize mainWindowSize = mainwin->size();
    //// 设置中心窗口大小为与主窗口大小一致
    //centralWidget->resize(mainWindowSize);

    // 获取中心窗口的大小
    QSize centralWindowSize = centralWidget->size();

    // 计算图像控件的大小
    int imageWidth = centralWindowSize.width();
    int imageHeight = centralWindowSize.height() * 0.8;

    //创建图像控件
    imageLabel = new QLabel(centralWidget);
    // 设置图像控件的大小
    imageLabel->setFixedSize(2100, 1200);
    imageLayout->addWidget(imageLabel, 0, Qt::AlignCenter | Qt::AlignHCenter);

    // 计算控件容器的大小
    int conWidth = centralWindowSize.width();
    int conHeight = centralWindowSize.height() * 0.2;
    //控件容器
    controlContainer = new QWidget(centralWidget);
    controlContainer->setFixedSize(2100, 100);
    controlLayout = new QHBoxLayout(controlContainer);
    controlContainer->setLayout(controlLayout);
    //layout->addWidget(controlContainer);

    // 将布局管理器添加到中心部件的布局管理器中
    layout = new QVBoxLayout(centralWidget);
    layout->addLayout(imageLayout);
    layout->addWidget(controlContainer, 0, Qt::AlignCenter | Qt::AlignHCenter);
    centralWidget->setLayout(layout);
}
/*
函数作用:创建右键菜单
函数参数:1、
         2、
         3、
*/
void UI::right_clickMenu(QMainWindow* mainwin)
{
    // 创建菜单对象
    clickmenu = new QMenu(mainwin);

    // 添加菜单项
    //撤销
    QAction* revoke = clickmenu->addAction(" 撤销 ");
    connect(revoke, &QAction::triggered, this, &UI::openImage);
    //反撤销
    QAction* reverse = clickmenu->addAction(" 反撤销 ");
    connect(revoke, &QAction::triggered, this, &UI::openImage);
    //保存文件
    QAction* saveFile = clickmenu->addAction(" 保存文件 ");
    connect(revoke, &QAction::triggered, this, &UI::saveImage);

    // 将菜单附加到窗口或控件
    mainwin->setContextMenuPolicy(Qt::CustomContextMenu); // 设置自定义上下文菜单策略
    connect(mainwin, &QWidget::customContextMenuRequested, [this](const QPoint& pos) {
        clickmenu->exec(mapToGlobal(pos)); // 在指定位置显示菜单
        });
}