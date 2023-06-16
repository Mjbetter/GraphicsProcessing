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
	/* 主界面自适应全屏,同时添加自定义最小化，最大化，退出按钮,同时添加对菜单栏进行初始化 */
	void InitializesMenuBar(QMainWindow *mainWindow);
	void Create_upleader();
	void Create_leftleader(QMainWindow* mainWindow);
	void Create_mainwin(QMainWindow* mainWindow);
	void Create_();
private slots:
    void On_ClickedMenuActionGroup(QAction* action);
    void On_ClickedToolBarActionGroup(QAction* action);

private:
    QVBoxLayout* mainLayout;
    //    QVBoxLayout *mainWidgetLayout;

    //    QWidget *mainWidget;

    QPushButton* m_PushButton;
    QMenu* m_Menu;
    QToolBar* m_ToolBar;
    QActionGroup* m_MenuActionGroup;
    QActionGroup* m_ToolBarActionGroup;
    QAction* m_ActionMenu1;
    QAction* m_ActionMenu2;
    QAction* m_ActionToolBar1;
    QAction* m_ActionToolBar2;

};