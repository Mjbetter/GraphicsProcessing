#include "GraphicsProcessing.h"

GraphicsProcessing::GraphicsProcessing(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    UI *ui = new UI();
    /*将屏幕设置为自适应全屏同时初始化菜单栏*/
    ui->InitializesMenuBar(this);
}

GraphicsProcessing::~GraphicsProcessing()
{}
