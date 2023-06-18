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
#include <QDebug>
#include <QStandardItemModel>
#include <QLineEdit>

#include <QPainter>


/*
函数作用:获得全屏的窗口，可放大缩小
函数参数:1、mainwin：指向窗口的指针
*/
void UI::initmainwin(QMainWindow* mainwin)
{
    mainwin->setWindowState(Qt::WindowMaximized); // 将主窗口最大化
    mainwin->setWindowFlags(Qt::Window | Qt::WindowMinMaxButtonsHint | Qt::WindowCloseButtonHint); // 显示最小化、最大化和关闭按钮
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
    imagePath = QFileDialog::getOpenFileName(this, "Open Image", QString(), "Image Files (*.png *.jpg *.bmp)");
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

    controlLayout->addWidget(labelKernelSize);
    controlLayout->addWidget(sliderKernelSize);
    controlLayout->addWidget(labelKernelNum);
    controlLayout->addWidget(lineEdit);


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


    //四级菜单
    // 细节处理的四级菜单
    //ImaDetail_ImaNoiPro_ImaNoiRe
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

/*
函数作用:创建右键菜单
函数参数:1、
         2、
         3、
*/
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
