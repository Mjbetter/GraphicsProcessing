#include "GraphicsProcessing.h"
#include <QMessageBox>
#include <QFileDialog>



//´´½¨
GraphicsProcessing::GraphicsProcessing(QMainWindow* parent) : QMainWindow(parent)
{
    UI* ui = new UI();
    ui->initmainwin(this);
}

// Ïú»Ù
GraphicsProcessing::~GraphicsProcessing()
{

}