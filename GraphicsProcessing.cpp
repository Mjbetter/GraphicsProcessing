#include "GraphicsProcessing.h"
#include <qmessagebox.h>
#include <qfiledialog.h>
#include "ui_GraphicsProcessing.h"


//´´½¨
GraphicsProcessing::GraphicsProcessing(QWidget* parent)
	:QMainWindow(parent)
{
	UI* ui = new UI();
	ui->initmainwin(this);
}

// Ïú»Ù
GraphicsProcessing::~GraphicsProcessing()
{

}