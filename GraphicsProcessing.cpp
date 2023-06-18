#include "GraphicsProcessing.h"
#include <QMessageBox>
#include <QFileDialog>



GraphicsProcessing::GraphicsProcessing(QMainWindow* parent) : QMainWindow(parent)
{
    UI* ui = new UI();
    ui->initmainwin(this);
}

GraphicsProcessing::~GraphicsProcessing()
{

}