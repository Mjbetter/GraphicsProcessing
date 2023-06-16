#ifndef GRAPHICSPROCESSING_H
#define GRAPHICSPROCESSING_H
// 通过结合使用这两个预处理指令，可以保证在同一个编译单元中
//（通常是一个源文件）只包含一次头文件，避免了重复定义和编译错误。

#include <QMainWindow>
#include "UI.h"
#include "ui_GraphicsProcessing.h"

class GraphicsProcessing : public QMainWindow
{
	Q_OBJECT
public:
	GraphicsProcessing(QMainWindow* parent = nullptr);
	~GraphicsProcessing();
private slots:

private:

};

#endif // !GRAPHICSPROCESSING.H
