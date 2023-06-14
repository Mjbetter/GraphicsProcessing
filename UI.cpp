#pragma
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
/*
函数作用:获得全屏的窗口，可放大缩小
函数参数:1、mainwin：指向窗口的指针
*/
void UI::initmainwin(QMainWindow* mainwin)
{
	//mainwin->resize(QSize(2000,1000));
    mainwin->showMaximized();
    createMenu(mainwin);
    createToolbar(mainwin);
    createCenterWin(mainwin);
}
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
        //它使用选定的文件路径创建一个名为image的QPixmap对象。
        QPixmap image(imagePath);
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
            imageLabel->setPixmap(image);
            imageLabel->adjustSize();        //调整图像标签的大小以适应加载的图像的大小。
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
函数参数:1、filePath：获取当前文件路径
         2、image：一个QPixmap类的对象
         3、imageLabel：将image设置成图像标签的像素图
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

    ImaCom = new QMenu(" 图像合成 ", menu);
    menu->addMenu(ImaCom);

    ImaSeg = new QMenu(" 图像分割 ", menu);
    menu->addMenu(ImaSeg);

    ImaNumRec = new QMenu(" 图像数字识别 ", menu);
    menu->addMenu(ImaNumRec);

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
    ImaZoomAction = new QAction(" 缩放 ", this);
    ImaAdjust->addAction(ImaZoomAction);
    ImaRotAction = new QAction(" 旋转 ", this);
    ImaAdjust->addAction(ImaRotAction);
    ImaMirrAction = new QAction(" 镜像 ", this);
    ImaAdjust->addAction(ImaMirrAction);

    //细节处理的二级菜单
    //边缘提取的二级菜单
    //图像处理的二级菜单
    //图像合成的二级菜单
    //图像分割的二级菜单
    //图像数字识别的二级菜单


}
/*
函数作用:设置上方工具栏
函数参数:1、toolBar
*/
void UI::createToolbar(QMainWindow* mainwin)
{
    toolBar = new QToolBar(mainwin);
    mainwin->addToolBar(Qt::LeftToolBarArea, toolBar);
    toolBar = addToolBar("File");
    toolBar->addAction(openAction);
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

    layout = new QVBoxLayout(centralWidget);
    imageLabel = new QLabel(centralWidget);
    layout->addWidget(imageLabel);
    layout->setAlignment(Qt::AlignCenter);
    
    QPushButton* openButton = new QPushButton("Open", centralWidget);
    connect(openButton, &QPushButton::clicked, this, &UI::openImage);
    layout->addWidget(openButton);

}