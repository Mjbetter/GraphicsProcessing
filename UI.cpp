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
<<<<<<< HEAD
#include <QDebug>
#include <QStandardItemModel>
#include <QLineEdit>

#include <QPainter>


=======
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
/*
函数作用:获得全屏的窗口，可放大缩小
函数参数:1、mainwin：指向窗口的指针
*/
void UI::initmainwin(QMainWindow* mainwin)
{
<<<<<<< HEAD
    mainwin->setWindowState(Qt::WindowMaximized); // 将主窗口最大化
    mainwin->setWindowFlags(Qt::Window | Qt::WindowMinMaxButtonsHint | Qt::WindowCloseButtonHint); // 显示最小化、最大化和关闭按钮
=======
    mainwin->setWindowState(Qt::WindowMaximized);
    createMenu(mainwin);
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
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
<<<<<<< HEAD
//根据索引寻找要实现的槽函数
void UI::handleMenuItemClicked(const QModelIndex& index)
{
    QStandardItemModel* model = qobject_cast<QStandardItemModel*>(this->treeView->model());
    QStandardItem* item = model->itemFromIndex(index);
    QString itemName = item->text();
    qDebug() << "Menu item clicked: " << itemName;

    // 根据菜单项的名称执行相应功能
    if (itemName == " 打开文件 ") {
        openImage();
    }
    else if (itemName == " 保存当前文件 ") {
        saveImage();
    }
    else if (itemName == " 查看当前文件信息 ") {
        showImageInfo();
    }
    else if (itemName == " 平移 ") {
        panImage();
    }
    else if (itemName == " 缩放 ") {
        zoomImage();
    }
    else if (itemName == " 旋转 ") {
        rotataImage();
    }
    else if (itemName == " 镜像 ") {
        mirrorImage();
    }
    else if (itemName == " 灰度图 ") {
        GrayImage();
    }
    else if (itemName == " 2值图 ") {
        BinaryImage();
    }
    else if (itemName == " 均值滤波 ") {
        MeanF();
    }
    else if (itemName == " 中值滤波 ") {
        MedianF();
    }
    else if (itemName == " 高斯滤波 ") {
        GaussianF();
    }
    else if (itemName == " 双边滤波 ") {
        BilateralF();
    }
    else if (itemName == " 小波滤波 ") {
        WaveletF();
    }
    else if (itemName == " 高斯噪声 ") {
        GaussianN();
    }
    else if (itemName == " 椒盐噪声 ") {
        SaltAndPepperN();
    }
    else if (itemName == " 泊松噪声 ") {
        PoissonN();
    }
    else if (itemName == " 边缘钝化 ") {
        BluntE();
    }
    else if (itemName == " 边缘锐化 ") {
        SharpE();
    }
    else if (itemName == " Roberts算子 ") {
        RobertsE();
    }
    else if (itemName == " Sobel算子 ") {
        SobelE();
    }
    else if (itemName == " Prewitt算子 ") {
        PrewittE();
    }
    else if (itemName == " Kirsch算子 ") {
        KirschE();
    }
    else if (itemName == " Robinsom算子 ") {
        RobinsomE();
    }
    else if (itemName == " Laplacian算子 ") {
        LaplacianE();
    }
    else if (itemName == " Canny算子 ") {
        CannyE();
    }
    else if (itemName == " 对比度增强 ") {
        ContrastE();
    }
    else if (itemName == " 亮度增强 ") {
        BrightnessE();
    }
    else if (itemName == " 直方图均衡化 ") {
        HistogramEqualization();
    }
    else if (itemName == " 指数变换增强 ") {
        ExponentialTransformationEnhancement();
    }
    else if (itemName == " 马赛克 ") {
        Mosaic();
    }
    else if (itemName == " 图像卷积 ") {
        ConvolutionImage();
    }
    else if (itemName == " 傅里叶变换 ") {
        FourierTransform();
    }
    else if (itemName == " 图像合成 ") {
        ImageSynthesis();
    }
    else if (itemName == " 图像分割 ") {
        ImageSegmentation();
    }
    else if (itemName == " 图像数字识别 ") {
        ImageDigitRecognition();
    }
    else if (itemName == " 其他 ") {

    }
}
/*
--------------------------------------Mat与QLabel的相互转换-------------------------------------------------------
----------------------------------通过判断通道数进行不同的转换----------------------------------------------------
=======
/*
--------------------------------------Mat与QLabel的相互转换------------------------------------------
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
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
<<<<<<< HEAD

    if (mat.channels() == 1) {
        // 单通道图像（灰度图像）
        cv::Mat grayImage;
        cv::cvtColor(mat, grayImage, cv::COLOR_GRAY2RGB); // 将单通道图像转换为三通道（RGB）图像

        QImage conimage(grayImage.data, grayImage.cols, grayImage.rows, QImage::Format_RGB888);
        QSize imageSize = conImalabel->size();
        QImage scaledImage = conimage.scaled(imageSize, Qt::KeepAspectRatio);
        conImalabel->setPixmap(QPixmap::fromImage(scaledImage));
    }
    else if (mat.channels() == 3) {
        // 三通道图像（彩色图像）
        cv::Mat rgbImage;
        cv::cvtColor(mat, rgbImage, cv::COLOR_BGR2RGB); // 将 BGR 图像转换为 RGB 图像

        QImage conimage(rgbImage.data, rgbImage.cols, rgbImage.rows, QImage::Format_RGB888);
        QSize imageSize = conImalabel->size();
        QImage scaledImage = conimage.scaled(imageSize, Qt::KeepAspectRatio);
        conImalabel->setPixmap(QPixmap::fromImage(scaledImage));
    }
    else {
        conImalabel->setText("Unsupported Image Format");
    }

    return conImalabel;
}

=======
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
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
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

<<<<<<< HEAD
    // 判断图像的格式和通道数
    if (!image.isNull()) {
        if (image.format() == QImage::Format_RGB888) {
            // 三通道图像（RGB）
            mat = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine()).clone();
            cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR); // 将 RGB 图像转换为 BGR 图像
        }
        else if (image.format() == QImage::Format_Indexed8) {
            // 单通道图像（灰度图像）
            mat = cv::Mat(image.height(), image.width(), CV_8UC1, const_cast<uchar*>(image.bits()), image.bytesPerLine()).clone();
        }
        else {
            qDebug() << "Unsupported Image Format";
        }
=======
    // 将 QImage 转换为 OpenCV 的 Mat
    if (!image.isNull()) {
        QImage rgbImage = image.convertToFormat(QImage::Format_RGB888);
        mat = cv::Mat(rgbImage.height(), rgbImage.width(), CV_8UC3, rgbImage.bits(), rgbImage.bytesPerLine()).clone();
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
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
<<<<<<< HEAD
    QSlider* sliderX = new QSlider(Qt::Horizontal, this); // 为X创建水平滚动条
    QSlider* sliderY = new QSlider(Qt::Horizontal, this); // 为Y创建水平滚动条
    //X
    sliderX->setMinimum(-100); // 设置最小值
    sliderX->setMaximum(100); // 设置最大值
    sliderX->setValue(0); // 设置初始值
    sliderX->setSingleStep(1); // 设置步长
    //Y
    sliderY->setMinimum(-100); // 设置最小值
    sliderY->setMaximum(100); // 设置最大值
    sliderY->setValue(0); // 设置初始值
    sliderY->setSingleStep(1); // 设置步长

    QLabel* labelx = new QLabel("X平移量:" + QString::number(sliderX->value()), this);
    QLabel* labely = new QLabel("Y平移量：" + QString::number(sliderY->value()), this);


    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(labelx);
    controlLayout->addWidget(sliderX);
    controlLayout->addWidget(labely);
    controlLayout->addWidget(sliderY);


    connect(sliderX, &QSlider::valueChanged, this, [this, labelx](int value) {
        xNum = value;
        labelx->setText("X平移量：" + QString::number(value));
        translateImage(xNum, yNum);//需修改
        });

    connect(sliderY, &QSlider::valueChanged, this, [this, labely](int value) {
        yNum = value;
        labely->setText("Y平移量：" + QString::number(value));
        translateImage(xNum, yNum);//需修改
        });

    translateImage(xNum, yNum);//需修改
}
//-----------------------------------------------------------------------------------------------------------------------------
void UI::translateImage(int dx, int dy)//需修改
{
    QImage testimage = imageLabel->pixmap()->toImage(); // 获取当前图像
    QPixmap testpixmap = QPixmap::fromImage(testimage); // 将图像转换为 pixmap

    // 创建一个平移后的图像副本
    QImage translatedImage(testpixmap.width(), testpixmap.height(), QImage::Format_RGB32);
    translatedImage.fill(Qt::black); // 使用黑色填充作为背景色

    // 执行平移操作
    QPainter painter(&translatedImage);
    painter.drawImage(dx, dy, testimage);
    painter.end();

    // 更新图像显示
    QPixmap translatedPixmap = QPixmap::fromImage(translatedImage);
    imageLabel->setPixmap(translatedPixmap);
}
//----------------------------------------------------------------------------------------------------------------------------
/*
函数作用：通过窗口获取放大或缩小的数值来进行图像缩放
函数参数：
*/
void UI::zoomImage()
{
    
    QSlider* sliderZoom = new QSlider(Qt::Horizontal, this); // 为zoomNum创建水平滚动条
    //zoomNum
    sliderZoom->setMinimum(-100); // 设置最小值
    sliderZoom->setMaximum(100); // 设置最大值
    sliderZoom->setValue(0); // 设置初始值
    sliderZoom->setSingleStep(1); // 设置步长
    QLabel* labelZoom = new QLabel("缩放量：" + QString::number(sliderZoom->value()), this);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderZoom);
    controlLayout->addWidget(labelZoom);
    connect(sliderZoom, &QSlider::valueChanged, this, [this,labelZoom](int value) {
        zoomNum = value;
        labelZoom->setText("缩放量：" + QString::number(value));
        //缩放函数位置
        });
    //缩放函数位置
}
/*
函数作用：通过窗口获取旋转的数值来进行图像旋转
函数参数：
*/
void UI::rotataImage()
{

    QSlider* sliderRotata = new QSlider(Qt::Horizontal, this); // 为zoomNum创建水平滚动条
    //zoomNum
    sliderRotata->setMinimum(-180); // 设置最小值
    sliderRotata->setMaximum(180); // 设置最大值
    sliderRotata->setValue(0); // 设置初始值
    sliderRotata->setSingleStep(1); // 设置步长
    QLabel* labelRotata = new QLabel("旋转角度：" + QString::number(sliderRotata->value()), this);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderRotata);
    controlLayout->addWidget(labelRotata);
    connect(sliderRotata, &QSlider::valueChanged, this, [this,labelRotata](int value) {
        rotataNum = value;
        labelRotata->setText("旋转角度：" + QString::number(value));
        //旋转函数位置
        });
    //旋转函数位置
}
/*
函数作用：直接调用函数进行图像镜像变换
函数参数：
*/
void UI::mirrorImage()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
=======
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
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00

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
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    QSlider* sliderCon_Kernel = new QSlider(Qt::Horizontal, this); // 为Con_KernelNum创建水平滚动条
    //zoomNum
    sliderCon_Kernel->setMinimum(0); // 设置最小值
    sliderCon_Kernel->setMaximum(100); // 设置最大值
    sliderCon_Kernel->setValue(0); // 设置初始值
    sliderCon_Kernel->setSingleStep(1); // 设置步长
    QLabel* labelCon_Kernel = new QLabel("卷积核大小：" + QString::number(sliderCon_Kernel->value()), this);
    Con_KernelSize = sliderCon_Kernel->value();

    controlLayout->addWidget(sliderCon_Kernel);
    controlLayout->addWidget(labelCon_Kernel);
    connect(sliderCon_Kernel, &QSlider::valueChanged, this, [this,labelCon_Kernel](int value) {
        Con_KernelSize = value;
        labelCon_Kernel->setText("卷积核大小：" + QString::number(value));
        //函数
        });
    //函数
}
//中值滤波
/*
函数作用：
函数参数：
*/
void UI::MedianF()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    QSlider* sliderCon_Kernel = new QSlider(Qt::Horizontal, this); // 为Con_KernelNum创建水平滚动条
    //zoomNum
    sliderCon_Kernel->setMinimum(0); // 设置最小值
    sliderCon_Kernel->setMaximum(100); // 设置最大值
    sliderCon_Kernel->setValue(0); // 设置初始值
    sliderCon_Kernel->setSingleStep(1); // 设置步长
    QLabel* labelCon_Kernel = new QLabel("卷积核大小：" + QString::number(sliderCon_Kernel->value()), this);
    Con_KernelSize = sliderCon_Kernel->value();

    controlLayout->addWidget(sliderCon_Kernel);
    controlLayout->addWidget(labelCon_Kernel);
    connect(sliderCon_Kernel, &QSlider::valueChanged, this, [this, labelCon_Kernel](int value) {
        Con_KernelSize = value;
        labelCon_Kernel->setText("卷积核大小：" + QString::number(value));
        //函数
        });
    //函数
}
//高斯滤波
/*
函数作用：
函数参数：
*/
void UI::GaussianF()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    QSlider* sliderCon_Kernel = new QSlider(Qt::Horizontal, this); // 为Con_KernelNum创建水平滚动条
    //zoomNum
    sliderCon_Kernel->setMinimum(0); // 设置最小值
    sliderCon_Kernel->setMaximum(100); // 设置最大值
    sliderCon_Kernel->setValue(0); // 设置初始值
    sliderCon_Kernel->setSingleStep(1); // 设置步长
    QLabel* labelCon_Kernel = new QLabel("卷积核大小：" + QString::number(sliderCon_Kernel->value()), this);
    Con_KernelSize = sliderCon_Kernel->value();

    controlLayout->addWidget(sliderCon_Kernel);
    controlLayout->addWidget(labelCon_Kernel);
    connect(sliderCon_Kernel, &QSlider::valueChanged, this, [this, labelCon_Kernel](int value) {
        Con_KernelSize = value;
        labelCon_Kernel->setText("卷积核大小：" + QString::number(value));
        //函数
        });
    //函数
}
//双边滤波
/*
函数作用：
函数参数：
*/
void UI::BilateralF()
{

<<<<<<< HEAD
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    QSlider* sliderCon_Kernel = new QSlider(Qt::Horizontal, this); // 为Con_KernelNum创建水平滚动条
    //zoomNum
    sliderCon_Kernel->setMinimum(0); // 设置最小值
    sliderCon_Kernel->setMaximum(100); // 设置最大值
    sliderCon_Kernel->setValue(0); // 设置初始值
    sliderCon_Kernel->setSingleStep(1); // 设置步长
    QLabel* labelCon_Kernel = new QLabel("卷积核大小：" + QString::number(sliderCon_Kernel->value()), this);
    Con_KernelSize = sliderCon_Kernel->value();

    controlLayout->addWidget(sliderCon_Kernel);
    controlLayout->addWidget(labelCon_Kernel);
    connect(sliderCon_Kernel, &QSlider::valueChanged, this, [this, labelCon_Kernel](int value) {
        Con_KernelSize = value;
        labelCon_Kernel->setText("卷积核大小：" + QString::number(value));
        //函数
        });
    //函数
}
//小波滤波
/*
函数作用：
函数参数：
*/
void UI::WaveletF()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
=======
    ImaCom = new QAction(" 图像合成 ", this);
    menu->addAction(ImaCom);
    connect(ImaCom, &QAction::triggered, this, &UI::ImageSynthesis);

    ImaSeg = new QAction(" 图像分割 ", this);
    menu->addAction(ImaSeg);
    connect(ImaSeg, &QAction::triggered, this, &UI::ImageSegmentation);

    ImaNumRec = new QAction(" 图像数字识别 ", this);
    menu->addAction(ImaNumRec);
    connect(ImaNumRec, &QAction::triggered, this, &UI::ImageDigitRecognition);
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00

    QSlider* sliderCon_Kernel = new QSlider(Qt::Horizontal, this); // 为Con_KernelNum创建水平滚动条
    //zoomNum
    sliderCon_Kernel->setMinimum(0); // 设置最小值
    sliderCon_Kernel->setMaximum(100); // 设置最大值
    sliderCon_Kernel->setValue(0); // 设置初始值
    sliderCon_Kernel->setSingleStep(1); // 设置步长
    QLabel* labelCon_Kernel = new QLabel("卷积核大小：" + QString::number(sliderCon_Kernel->value()), this);
    Con_KernelSize = sliderCon_Kernel->value();

<<<<<<< HEAD
    controlLayout->addWidget(sliderCon_Kernel);
    controlLayout->addWidget(labelCon_Kernel);
    connect(sliderCon_Kernel, &QSlider::valueChanged, this, [this, labelCon_Kernel](int value) {
        Con_KernelSize = value;
        labelCon_Kernel->setText("卷积核大小：" + QString::number(value));
        //函数
        });
    //函数
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
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//Sobel算子
/*
函数作用：
函数参数：
*/
void UI::SobelE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//Prewitt算子
/*
函数作用：
函数参数：
*/
void UI::PrewittE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//Kirsch算子
/*
函数作用：
函数参数：
*/
void UI::KirschE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//Robinsom算子
/*
函数作用：
函数参数：
*/
void UI::RobinsomE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//_Laplacian算子
void  UI::LaplacianE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//_Canny算子
void UI::CannyE()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}

/*
----------------------------------------------------图像处理------------------------------------------------
*/
//图像增强
//_对比度增强
/*
函数作用：对比度增强
函数参数：
*/
void UI::ContrastE()
{

    QSlider* sliderContrast = new QSlider(Qt::Horizontal, this); // 为ContrastNum创建水平滚动条
    //ContrastNum
    sliderContrast->setMinimum(0); // 设置最小值
    sliderContrast->setMaximum(100); // 设置最大值
    sliderContrast->setValue(0); // 设置初始值
    sliderContrast->setSingleStep(1); // 设置步长
    QLabel* labelRotata = new QLabel("对比增强：" + QString::number(sliderContrast->value()), this);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderContrast);
    controlLayout->addWidget(labelRotata);
    connect(sliderContrast, &QSlider::valueChanged, this, [this,labelRotata](int value) {
        ContrastNum = value;
        labelRotata->setText("对比增强：" + QString::number(value));
        //函数
        });
    //函数
}
//_亮度增强
/*
函数作用：
函数参数：
*/
void UI::BrightnessE()
{

    QSlider* sliderBrightness = new QSlider(Qt::Horizontal, this); // 为BrightnessNum创建水平滚动条
    //ContrastNum
    sliderBrightness->setMinimum(-100); // 设置最小值
    sliderBrightness->setMaximum(100); // 设置最大值
    sliderBrightness->setValue(0); // 设置初始值
    sliderBrightness->setSingleStep(1); // 设置步长
    QLabel* labelBrightness = new QLabel("亮度增强" + QString::number(sliderBrightness->value()), this);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderBrightness);
    controlLayout->addWidget(labelBrightness);
    connect(sliderBrightness, &QSlider::valueChanged, this, [this,labelBrightness](int value) {
        BrightnessNum = value;
        labelBrightness->setText("亮度增强" + QString::number(value));
        //函数
        });
    //函数
}
//_直方图均衡化
/*
函数作用：
函数参数：
*/
void UI::HistogramEqualization()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
//_指数变化增强
void UI::ExponentialTransformationEnhancement()
{
    QSlider* sliderETE1 = new QSlider(Qt::Horizontal, this); // 为Exponential1Num创建水平滚动条
//Exponential1Num
    sliderETE1->setMinimum(-100); // 设置最小值
    sliderETE1->setMaximum(100); // 设置最大值
    sliderETE1->setValue(0); // 设置初始值
    sliderETE1->setSingleStep(1); // 设置步长
    QLabel* labelETE1 = new QLabel("指数变化指标2：" + QString::number(sliderETE1->value()), this);

    QSlider* sliderETE2 = new QSlider(Qt::Horizontal, this); // 为Exponential2Num创建水平滚动条
//Exponential2Num
    sliderETE2->setMinimum(-100); // 设置最小值
    sliderETE2->setMaximum(100); // 设置最大值
    sliderETE2->setValue(0); // 设置初始值
    sliderETE2->setSingleStep(1); // 设置步长
    QLabel* labelETE2 = new QLabel("指数变化指标2：" + QString::number(sliderETE2->value()), this);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderETE1);
    controlLayout->addWidget(labelETE1);
    controlLayout->addWidget(sliderETE2);
    controlLayout->addWidget(labelETE2);
    connect(sliderETE1, &QSlider::valueChanged, this, [this, labelETE1](int value) {
        Exponential1Num = value;
        labelETE1->setText("指数变化指标1：" + QString::number(value));
        //函数
        });
    connect(sliderETE2, &QSlider::valueChanged, this, [this, labelETE2](int value) {
        Exponential1Num = value;
        labelETE2->setText("指数变化指标1：" + QString::number(value));
        //函数
        });
    //函数
}
//加马赛克
/*
函数作用：
函数参数：
*/
void UI::Mosaic()
{


    QSlider* sliderMosaic = new QSlider(Qt::Horizontal, this); // 为MosaicNum创建水平滚动条
    //ContrastNum
    sliderMosaic->setMinimum(0); // 设置最小值
    sliderMosaic->setMaximum(50); // 设置最大值
    sliderMosaic->setValue(0); // 设置初始值
    sliderMosaic->setSingleStep(1); // 设置步长
    QLabel* labelMosaic = new QLabel("马赛克程度" + QString::number(sliderMosaic->value()), this);

    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);

    controlLayout->addWidget(sliderMosaic);
    controlLayout->addWidget(labelMosaic);
    connect(sliderMosaic, &QSlider::valueChanged, this, [this,labelMosaic](int value) {
        MosaicNum = value;
        labelMosaic->setText("马赛克程度" + QString::number(value));
        });
}
//图像卷积
/*
函数作用：
函数参数：
*/
void UI::ConvolutionImage()
{

    QSlider* sliderKernelSize = new QSlider(Qt::Horizontal, this); // 为KernelSize创建水平滚动条
    //KernelSize
    sliderKernelSize->setMinimum(0); // 设置最小值
    sliderKernelSize->setMaximum(10); // 设置最大值
    sliderKernelSize->setValue(0); // 设置初始值
    sliderKernelSize->setSingleStep(1); // 设置步长
    QLabel* labelKernelSize = new QLabel("卷积核大小：" + QString::number(sliderKernelSize->value()), this);
    
    //KernelNum
    QLineEdit* lineEdit = new QLineEdit(this); // 创建一个单行文本框控件
    // 将输入限制为整数
    QIntValidator* validator = new QIntValidator(this);
    lineEdit->setValidator(validator);
    QLabel* labelKernelNum = new QLabel("卷积核数量：");
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
=======
    //二级菜单

    //文件的二级菜单
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00

    controlLayout->addWidget(labelKernelSize);
    controlLayout->addWidget(sliderKernelSize);
    controlLayout->addWidget(labelKernelNum);
    controlLayout->addWidget(lineEdit);


<<<<<<< HEAD
    connect(sliderKernelSize, &QSlider::valueChanged, this, [this,labelKernelSize](int value) {
        KernelSize = value;
        labelKernelSize->setText("卷积核大小：" + QString::number(value));
        //函数
        });
    // 连接文本框的textChanged信号与槽函数进行处理,当回车时文本传输
    connect(lineEdit, &QLineEdit::returnPressed, this, [this,lineEdit]() {
        QString inputText = lineEdit->text();
        KernelNum = inputText.toInt();
        // 后续的函数
        });
    //函数
}
//傅里叶变换
/*
函数作用：
函数参数：
*/
void UI::FourierTransform()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
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
            QImage rgbaOtherImage = Otherimage.convertToFormat(QImage::Format_RGBA8888);
            cv::Mat Othermat(rgbaOtherImage.height(), rgbaOtherImage.width(), CV_8UC4, rgbaOtherImage.bits(), rgbaOtherImage.bytesPerLine());
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
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
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
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
}
/*
---------------------------------------------图像素描化-------------------------------------------------------------------
*/
/*
函数作用：
函数参数：
*/
void UI::ImageSketching()
{
    //删除空间变换区域原有控件
    deleteChildWidgets(controlContainer);
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
函数作用:设置上方菜单栏
函数参数:1、fileMenu:
         2、openAction：
*/
QStandardItemModel* UI::createLeftMenu(QWidget* leftwidget)
{
    QStandardItemModel* menumodel = new QStandardItemModel();
    QStandardItem* rootItem = menumodel->invisibleRootItem(); // 获取根项
    //一级菜单

    fileOP = new QStandardItem(" 文件 ");
    menumodel->insertRow(0, fileOP);

    ImaAdjust = new QStandardItem(" 图像调整 ");
    menumodel->insertRow(1, ImaAdjust);

    ImaDetail = new QStandardItem(" 细节处理 ");
    menumodel->insertRow(2, ImaDetail);

    ImaEdge = new QStandardItem(" 边缘提取 ");
    menumodel->insertRow(3, ImaEdge);

    ImaPro = new QStandardItem(" 图像处理 ");
    menumodel->insertRow(4, ImaPro);

    ImaCom = new QStandardItem(" 图像合成 ");
    menumodel->insertRow(5, ImaCom);

    ImaSeg = new QStandardItem(" 图像分割 ");
    menumodel->insertRow(6, ImaSeg);

    ImaNumRec = new QStandardItem(" 图像数字识别 ");
    menumodel->insertRow(7, ImaNumRec);

    ImaSketching = new QStandardItem(" 图像素描化 ");
    menumodel->insertRow(8, ImaSketching);

    //二级菜单

    //文件的二级菜单

    openAction = new QStandardItem(" 打开文件 ");
    fileOP->appendRow(openAction);

    saveAction = new QStandardItem(" 保存当前文件 ");
    fileOP->appendRow(saveAction);

    vImaInfoAction = new QStandardItem(" 查看当前文件信息 ");
    fileOP->appendRow(vImaInfoAction);

    //图像调整的二级菜单

    ImaPanAction = new QStandardItem(" 平移 ");
    ImaAdjust->appendRow(ImaPanAction);

    ImaZoomAction = new QStandardItem(" 缩放 ");
    ImaAdjust->appendRow(ImaZoomAction);

    ImaRotAction = new QStandardItem(" 旋转 ");
    ImaAdjust->appendRow(ImaRotAction);

    ImaMirrAction = new QStandardItem(" 镜像 ");
    ImaAdjust->appendRow(ImaMirrAction);

    //细节处理的二级菜单
    ImaGS = new QStandardItem(" 变灰度 ");
    ImaDetail->appendRow(ImaGS);

    ImaNoiPro = new QStandardItem(" 噪声处理 ");
    ImaDetail->appendRow(ImaNoiPro);

    ImaPasEdg = new QStandardItem(" 边缘钝化 ");
    ImaDetail->appendRow(ImaPasEdg);

    ImaShrEdg = new QStandardItem(" 边缘锐化 ");
    ImaDetail->appendRow(ImaShrEdg);

    //边缘提取的二级菜单
    ImaFirRober = new QStandardItem(" Roberts算子 ");
    ImaEdge->appendRow(ImaFirRober);

    ImaFirSobel = new QStandardItem(" Sobel算子 ");
    ImaEdge->appendRow(ImaFirSobel);

    ImaFirPrewi = new QStandardItem(" Prewitt算子 ");
    ImaEdge->appendRow(ImaFirPrewi);

    ImaFirKirsc = new QStandardItem(" Kirsch算子 ");
    ImaEdge->appendRow(ImaFirKirsc);

    ImaFirRobin = new QStandardItem(" Robinsom算子 ");
    ImaEdge->appendRow(ImaFirRobin);
    QStandardItem* ImaFirCanny;

    ImaFirLapla = new QStandardItem(" Laplacian算子 ");
    ImaEdge->appendRow(ImaFirLapla);

    ImaFirCanny = new QStandardItem(" Canny算子 ");
    ImaEdge->appendRow(ImaFirCanny);


    //图像处理的二级菜单
    ImaEnh = new QStandardItem(" 图像增强 ");
    ImaPro->appendRow(ImaEnh);

    ImaMos = new QStandardItem(" 马赛克 ");
    ImaPro->appendRow(ImaMos);

    ImaConv = new QStandardItem(" 图像卷积 ");
    ImaPro->appendRow(ImaConv);

    ImaFourAnal = new QStandardItem(" 傅里叶变换 ");
    ImaPro->appendRow(ImaFourAnal);

    //三级菜单

    //细节处理的三级菜单
    //ImaDetail_ImaGS
    ImaGray = new QStandardItem(" 灰度图 ");
    ImaGS->appendRow(ImaGray);

    ImaBin = new QStandardItem(" 2值图 ");
    ImaGS->appendRow(ImaBin);

    //ImaDetail_ImaNoiPro
    ImaNoiRe = new QStandardItem(" 去噪 ");
    ImaNoiPro->appendRow(ImaNoiRe);

    ImaNoiAdd = new QStandardItem(" 加噪 ");
    ImaNoiPro->appendRow(ImaNoiAdd);

    //图像处理的三级菜单
    //ImaPro_ImaEnh
    ImaCE = new QStandardItem(" 对比度增强 ");
    ImaEnh->appendRow(ImaCE);

    ImaBE = new QStandardItem(" 亮度增强 ");
    ImaEnh->appendRow(ImaBE);

    ImaHE = new QStandardItem(" 直方图均衡化 ");
    ImaEnh->appendRow(ImaHE);

    ImaETE = new QStandardItem(" 指数变换增强 ");
    ImaEnh->appendRow(ImaETE);

=======
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
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00

    //四级菜单
    // 细节处理的四级菜单
    //ImaDetail_ImaNoiPro_ImaNoiRe
<<<<<<< HEAD
    ImaMeanF = new QStandardItem(" 均值滤波 ");
    ImaNoiRe->appendRow(ImaMeanF);

    ImaMediF = new QStandardItem(" 中值滤波 ");
    ImaNoiRe->appendRow(ImaMediF);

    ImaGausF = new QStandardItem(" 高斯滤波 ");
    ImaNoiRe->appendRow(ImaGausF);

    ImaBiluF = new QStandardItem(" 双边滤波 ");
    ImaNoiRe->appendRow(ImaBiluF);

    ImaWaveF = new QStandardItem(" 小波滤波 ");
    ImaNoiRe->appendRow(ImaWaveF);

    //ImaDetail_ImaNoiPro_ImaNoiAdd;
    ImaGausN = new QStandardItem(" 高斯噪声 ");
    ImaNoiAdd->appendRow(ImaGausN);

    ImaSAPN = new QStandardItem(" 椒盐噪声 ");
    ImaNoiAdd->appendRow(ImaSAPN);

    ImaPoiN = new QStandardItem(" 泊松噪声 ");
    ImaNoiAdd->appendRow(ImaPoiN);





    // 设置菜单的样式表


    return menumodel;

=======
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
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
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
<<<<<<< HEAD
    //给中心窗口设置布局
    mainlayout = new QHBoxLayout(centralWidget);

    //将中心窗口设置成左右两个布局
    leftWidgetLayout = new QVBoxLayout();
    rightWidgetLayout = new QVBoxLayout();
    //将两个布局加入中心窗口的布局
    mainlayout->addLayout(leftWidgetLayout);
    mainlayout->addLayout(rightWidgetLayout);

    //将菜单放在左边布局
    menubarLayout = new QVBoxLayout();
    leftWidgetLayout->addLayout(menubarLayout);
    MenuModel = createLeftMenu(leftWidget);
    // 创建树视图
    treeView = new QTreeView();

    treeView->setModel(MenuModel);
    // 将树视图添加到布局中
    menubarLayout->addWidget(treeView);
    //将点击与槽函数相连
    connect(treeView, &QTreeView::clicked, this, &UI::handleMenuItemClicked);

    //将图像和控件容器放在右边布局

    // 创建图像控件及其布局管理器
    imageLabel = new QLabel();
    // 设置图像控件的大小
    imageLabel->setFixedSize(2100, 1200);
    imageLabel->setParent(rightWidget);
    imageLayout = new QVBoxLayout(imageLabel);
    rightWidgetLayout->addWidget(imageLabel,0, Qt::AlignCenter | Qt::AlignHCenter);


    //控件容器
    controlContainer = new QWidget(rightWidget);
    controlContainer->setFixedSize(2100, 100);
    controlLayout = new QHBoxLayout(controlContainer);
    controlContainer->setLayout(controlLayout);
    rightWidgetLayout->addWidget(controlContainer);


    centralWidget->setLayout(mainlayout);
}

=======
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
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
/*
函数作用:创建右键菜单
函数参数:1、
         2、
         3、
*/
<<<<<<< HEAD
void UI::right_clickMenu(QWidget* centralWidget)
{
    // 创建菜单对象
    clickmenu = new QMenu(centralWidget);
    // 设置菜单样式表
    QString menuStyle = "QMenu { background-color: #f0f0f0; border: 1px solid #707070; }"
        "QMenu::item { padding: 5px 30px 5px 20px; }"
        "QMenu::item:selected { background-color: #0078d7; color: #ffffff; }";
    clickmenu->setStyleSheet(menuStyle);
    // 添加菜单项
    // 撤销
    QAction* revoke = clickmenu->addAction(" 撤销 ");
    connect(revoke, &QAction::triggered, this, &UI::openImage);
    // 反撤销
    QAction* reverse = clickmenu->addAction(" 反撤销 ");
    connect(reverse, &QAction::triggered, this, &UI::openImage);
    // 保存文件
    QAction* saveFile = clickmenu->addAction(" 保存文件 ");
    connect(saveFile, &QAction::triggered, this, &UI::saveImage);

    // 将菜单附加到中心窗口部件
    centralWidget->setContextMenuPolicy(Qt::CustomContextMenu); // 设置自定义上下文菜单策略
    connect(centralWidget, &QWidget::customContextMenuRequested, [this, centralWidget](const QPoint& pos) {
        clickmenu->exec(centralWidget->mapToGlobal(pos)); // 在指定位置显示菜单
        });
}
=======
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
>>>>>>> 580a69773558cd4786d11b342ab803c2ab1f2a00
