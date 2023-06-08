#include "UI.h"


using namespace std;


/*
函数作用:让界面自适应屏幕大小，同时添加对菜单栏进行初始化
函数参数:1、mainWindow：指向主界面的指针，用于对主界面进行界面布局操作
*/
void UI::InitializesMenuBar(QMainWindow* mainWindow) {
	
	/*设置界面大小*/
	mainWindow->resize(1920, 1200);	
	
}