
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QToolBar>
#include <QLabel>
#include <QPixmap>
#include <qmenu.h>
#include <qmenubar.h>
#include <QVBoxLayout>
#include <QSplitter>

class UI : public QMainWindow
{

public:


	//设置主窗口
	void initmainwin(QMainWindow* mainwin);
	
	//设置菜单
	void createMenu(QMainWindow* mainwin);
	//设置工具栏
	void createToolbar(QMainWindow* mainwin);
	//设置中心窗口
	void createCenterWin(QMainWindow* mainwin);

	//上方菜单
	QMenuBar* menu;
	QMenu* fileOP;
	QMenu* ImaAdjust;
	QMenu* ImaDetail;
	QMenu* ImaEdge;
	QMenu* ImaPro;
	QMenu* ImaCom;
	QMenu* ImaSeg;
	QMenu* ImaNumRec;
	QMenu* ElseFunc;
	//fileOP
	QAction* openAction;
	QAction* saveAction;
	QAction* vImaInfoAction;
	//ImaAdjust
	QAction* ImaPanAction;
	QAction* ImaZoomAction;
	QAction* ImaRotAction;
	QAction* ImaMirrAction;


	QToolBar* toolBar;

	QLabel* imageLabel = nullptr;

	QString imagePath;
	QString savePath;

	QWidget* centralWidget;
	QVBoxLayout* layout;

private:

public	slots:
	//打开图像文件
	void openImage();
	//保存图像文件
	void saveImage();
	//查看图像信息
	void showImageInfo();
};