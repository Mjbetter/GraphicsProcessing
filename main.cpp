#include "GraphicsProcessing.h"
#include <QtWidgets/QApplication>

/*
命名规范：
类名：首字母大写，单词和单词之间首字母大写
函数名：变量名称 首字母小写，单词和单词之间首字母大写
*/

/*
注释规范：
注释统一当前格式来包括
注明函数作用
注明参数作用及意义
注明函数返回值
*/

/*
关于算法实现的封装，统一在Algorithm.h中声明函数，在Algorithm.cpp中实现函数
关于QT界面的封装，在Form Files中放置存储的自定义控件，在对应调用位置注释控件相关说明
关于QT界面的代码设置，如果需要用代码对界面进行优化，封装成函数在UI类中实现
关于资源文件的导入，统一放入到GraphicsProcessing.qrc中，资源文件名称统一由英文代替汉字，格式为.png格式
*/

/*
项目每天写完后用git上传至github中，下次在写就拉下来，VS中可以配置git对于项目的快速拉取推送
*/

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GraphicsProcessing w;
    w.show();
    return a.exec();
}
