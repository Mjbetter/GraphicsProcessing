#include "UI.h"

#include <QMenu>
#include <QAction>
#include <QMenuBar>
#include <QVBoxLayout>
#include <QToolBar>
#include <QWidget>
#include <QHBoxLayout>
#include <QActionGroup>
#include <QPushButton>
#include <QDebug>

using namespace std;

/*
	界面中的分项布局
*/

/*
函数作用:让界面自适应屏幕大小，同时添加对菜单栏进行初始化
函数参数:1、mainWindow：指向主界面的指针，用于对主界面进行界面布局操作
*/
void UI::InitializesMenuBar(QMainWindow* mainWindow) {
	
	/* 设置界面大小 */
	//mainWindow->resize(1920, 1200);	
	//在ui中可以直接设计
	/*
		最大化最小化以及关闭按钮
	*/
	mainWindow->setWindowFlags(Qt::Dialog | Qt::WindowMinMaxButtonsHint | Qt::WindowCloseButtonHint);
}
/*
函数作用:细节设置上方菜单栏
函数参数:1、uptb：指向主界面工具栏的指针，完成对上方工具栏的布局
*/

void UI::Create_upleader()
{	
	m_ToolBar = new QToolBar;
	m_ToolBarActionGroup = new QActionGroup(this);
	QList<QAction*> m_ToolBarList;

	m_ActionToolBar1 = new QAction(QStringLiteral("ToolBar1"));
	m_ActionToolBar2 = new QAction(QStringLiteral("ToolBar2"));
	m_ActionToolBar1->setCheckable(true);
	m_ActionToolBar2->setCheckable(true);
	m_ToolBarList.append(m_ActionToolBar1);
	m_ToolBarList.append(m_ActionToolBar2);
	m_ToolBarActionGroup->addAction(m_ActionToolBar1);
	m_ToolBarActionGroup->addAction(m_ActionToolBar2);

	m_ToolBar->addActions(m_ToolBarList);
	m_ToolBar->setStyleSheet("background-color:rgb(200,40,43);color:rgb(204,204,204)");
	mainLayout->addWidget(m_ToolBar);
}

void UI::Create_leftleader(QMainWindow* mainWindow)
{

	//QToolBar* toolBarPointer = new QToolBar(this);		//创建QToolBar对象
	//this->addToolBar(Qt::LeftToolBarArea, toolBarPointer);		//将该对象添加到主窗口上，并设置默认停靠在左侧
	//toolBarPointer->setMovable(true);		//设置该对象为可移动
	//toolBarPointer->setFloatable(false);		//设置该对象不可悬浮
	//toolBarPointer->setAllowedAreas(Qt::LeftToolBarArea | Qt::RightToolBarArea);		//设置该对象可停靠的区域为左侧和右侧

	//QMenu* menuTest = new QMenu("测试1", this);		//创建一个menu对象
	//QAction* actionWelcome = menuTest->addAction("欢迎");		//在menu对象上添加action
	//QMenu* menuWelcome = new QMenu("测试2", this);		//创建第二个menu对象
	//actionWelcome->setMenu(menuWelcome);		//将第二个menu对象依附于action
	//menuWelcome->addAction("项目");			//在action上添加action“项目”
	//menuWelcome->addAction("示例");			//在action上添加action“示例”
	//menuWelcome->addAction("教程");			//在action上添加action“教程”
	//toolBarPointer->addAction(actionWelcome);		//将action对象依附于QToolBar


	////Menu的格式
	////"QMenuBar{background-color:transparent;}"/*设置背景色，跟随背景色*/
	////"QMenuBar::selected{background-color:transparent;}"/*设置菜单栏选中背景色*/
	////"QMenuBar::item{font-size:12px;font-family:Microsoft YaHei;color:rgba(255,255,255,1);}"/*设置菜单栏字体为白色，透明度为1（取值范围0.0-255）*/

	////创建菜单栏
	//QMenuBar* menubar = new QMenuBar(mainWindow);
	////添加菜单栏到主窗口中
	//mainWindow->setMenuBar(menubar);
	////创建菜单
	//QMenu* menu1 = new QMenu("第七组", mainWindow);
	//QMenu* menu2 = new QMenu("文件", mainWindow);
	//QMenu* menu3 = new QMenu("撤销", mainWindow);
	//QMenu* menu4 = new QMenu("反撤销", mainWindow);
	//QMenu* menu5 = new QMenu("文件", mainWindow);
	//QMenu* menu6 = new QMenu("工具", mainWindow);
	//QMenu* menu7 = new QMenu("控件", mainWindow);
	//QMenu* menu8 = new QMenu("帮助", mainWindow);

	////菜单栏添加菜单
	//menubar->addMenu(menu1);
	//menubar->addMenu(menu2);
	//menubar->addMenu(menu3);
	//menubar->addMenu(menu4);
	//menubar->addMenu(menu5);
	//menubar->addMenu(menu6);
	//menubar->addMenu(menu7);
	//menubar->addMenu(menu8);

	////创建菜单项
	//QAction* action1 = new QAction("打开文件", mainWindow);
	//QAction* action2 = new QAction("2", mainWindow);
	//QAction* action3 = new QAction("退出", mainWindow);

	////菜单添加菜单项
	//menu1->addAction(action1);
	//menu1->addAction(action2);
	//menu1->addSeparator();//插入分割线

	//QMenu* menu9 = new QMenu("最近访问的文件", mainWindow);
	//menu1->addMenu(menu9);//添加二级菜单
	//menu9->addAction(new QAction("暂无最近打开项目", mainWindow));//二级菜单添加菜单项

	//menu1->addAction(action3);
}

void UI::Create_mainwin(QMainWindow* mainWindow)
{

}

