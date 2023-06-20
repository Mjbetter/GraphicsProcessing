#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    char ad[128] = { 0 };
    int  filename = 0, filenum = 0;
    Mat img = imread("./digits.png");
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    int b = 20;
    int m = gray.rows / b;   //原图为1000*2000
    int n = gray.cols / b;   //裁剪为5000个20*20的小图块

    for (int i = 0; i < m; i++)
    {
        int offsetRow = i * b;  //行上的偏移量
        if (i % 5 == 0 && i != 0)
        {
            filename++;
            filenum = 0;
        }
        for (int j = 0; j < n; j++)
        {
            int offsetCol = j * b; //列上的偏移量
            sprintf_s(ad, "D:\\data\\%d\\%d.jpg", filename, filenum++);
            //截取20*20的小块
            Mat tmp;
            gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
            imwrite(ad, tmp);
        }
    }
    return 0;
}