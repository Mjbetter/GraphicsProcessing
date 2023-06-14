#pragma once
#include <qdesktopwidget.h>
#include <QtWidgets/QMainWindow>
#include "ui_GraphicsProcessing.h"
#include <QtWidgets/QApplication>
#include <QPushButton.h>
#include <QPixmap.h>
#include <QStyle.h>
#include <qdialog.h>
#include <QToolButton>
#include <qwidget.h>
#include <iostream>
#include <qaction.h>
#include <qlayout.h>
#include "GraphicsProcessing.h"
#include <qmenu.h>

class UI 
{
public:
	/*主界面自适应全屏,同时添加自定义最小化，最大化，退出按钮*/
	void InitializesMenuBar(QMainWindow *mainWindow);
};