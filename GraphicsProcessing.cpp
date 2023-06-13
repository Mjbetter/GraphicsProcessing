#include "GraphicsProcessing.h"

#include <QMenu>
#include <QAction>
#include <QMenuBar>
#include <QVBoxLayout>
/*
    主要界面设置
*/
GraphicsProcessing::GraphicsProcessing(QWidget* parent)
    : QMainWindow(parent)
{
    //ui.setupUi(this);
    //UI* ui = new UI();
    ///*将屏幕设置为自适应全屏同时初始化菜单栏*/
    //ui->InitializesMenuBar(this);
    setWindowTitle("Image Editor");
    setupUi();
}

void GraphicsProcessing::setupUi()
{
    // 创建主窗口布局
    mainLayout = new QHBoxLayout();
    QWidget* mainWidget = new QWidget();
    mainWidget->setLayout(mainLayout);
    setCentralWidget(mainWidget);

    // 创建左侧工具栏
    toolbarLayout = new QVBoxLayout();
    toolbarWidget = new QWidget();
    toolbarWidget->setLayout(toolbarLayout);
    mainLayout->addWidget(toolbarWidget);

    toolLabel = new QLabel("Tools:");
    toolbarLayout->addWidget(toolLabel);

    selectButton = new QPushButton(QIcon(":/icons/select.png"), "Select");
    toolbarLayout->addWidget(selectButton);

    drawButton = new QPushButton(QIcon(":/icons/draw.png"), "Draw");
    toolbarLayout->addWidget(drawButton);

    eraseButton = new QPushButton(QIcon(":/icons/erase.png"), "Erase");
    toolbarLayout->addWidget(eraseButton);

    fillButton = new QPushButton(QIcon(":/icons/fill.png"), "Fill");
    toolbarLayout->addWidget(fillButton);

    brushLabel = new QLabel("Brush Size:");
    toolbarLayout->addWidget(brushLabel);

    brushSizeSpinBox = new QSpinBox();
    brushSizeSpinBox->setRange(1, 100);
    brushSizeSpinBox->setValue(10);
    toolbarLayout->addWidget(brushSizeSpinBox);

    opacityLabel = new QLabel("Opacity:");
    toolbarLayout->addWidget(opacityLabel);

    opacitySlider = new QSlider(Qt::Horizontal);
    opacitySlider->setRange(0, 100);
    opacitySlider->setValue(100);
    toolbarLayout->addWidget(opacitySlider);

    colorLabel = new QLabel("Color:");
    toolbarLayout->addWidget(colorLabel);

    colorComboBox = new QComboBox();
    colorComboBox->addItem(QIcon(":/icons/color_black.png"), "Black");
    colorComboBox->addItem(QIcon(":/icons/color_red.png"), "Red");
    colorComboBox->addItem(QIcon(":/icons/color_green.png"), "Green");
    colorComboBox->addItem(QIcon(":/icons/color_blue.png"), "Blue");
    toolbarLayout->addWidget(colorComboBox);

    // 创建右侧图像显示区域
    imageLabel = new QLabel();
    QPixmap pixmap(":/images/example.png");
    imageLabel->setPixmap(pixmap);
    mainLayout->addWidget(imageLabel);
}