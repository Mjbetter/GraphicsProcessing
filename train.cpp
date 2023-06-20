#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>


using namespace std;
using namespace cv;
using namespace cv::ml;


//**自定义结构体
struct MyNum
{
	cv::Mat mat; //数字图片
	cv::Rect rect;//相对整张图所在矩形
	int label;//数字标签
};

int main()
{
	Mat src = imread("digit.png");
	if (src.empty())
	{
		cout << "No Image..." << endl;
		system("pause");
		return -1;
	}

	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	const int classNum = 10;  //总共有0~9个数字类别
	const int picNum = 20;//每个类别共20张图片
	const int pic_w = 28;//图片宽
	const int pic_h = 28;//图片高

	//将数据集分为训练集、测试集
	double totalNum = classNum * picNum;//图片总数
	double per = 0.8;	//百分比--修改百分比可改变训练集、测试集比重
	double trainNum = totalNum * per;//训练图片数量
	double testNum = totalNum * (1.0 - per);//测试图片数量

	Mat Train_Data, Train_Label;//用于训练
	vector<MyNum>TestData;//用于测试
	for (int i = 0; i < picNum; i++)
	{
		for (int j = 0; j < classNum; j++)
		{
			//将所有图片数据都拷贝到Mat矩阵里
			Mat temp;
			gray(Range(j * pic_w, j * pic_w + pic_w), Range(i * pic_h, i * pic_h + pic_h)).copyTo(temp);
			Train_Data.push_back(temp.reshape(0, 1)); //将temp数字图像reshape成一行数据，然后一一追加到Train_Data矩阵中
			Train_Label.push_back(j);

			//额外用于测试
			if (i * classNum + j >= trainNum)
			{
				TestData.push_back({ temp,Rect(i * pic_w,j * pic_h,pic_w,pic_h),j });
			}
		}
	}

	//准备训练数据集
	Train_Data.convertTo(Train_Data, CV_32FC1); //转化为CV_32FC1类型
	Train_Label.convertTo(Train_Label, CV_32FC1);
	Mat TrainDataMat = Train_Data(Range(0, trainNum), Range::all()); //只取trainNum行训练
	Mat TrainLabelMat = Train_Label(Range(0, trainNum), Range::all());

	//KNN训练
	const int k = 3;  //k值，取奇数，影响最终识别率
	Ptr<KNearest>knn = KNearest::create();  //构造KNN模型
	knn->setDefaultK(k);//设定k值
	knn->setIsClassifier(true);//KNN算法可用于分类、回归。
	knn->setAlgorithmType(KNearest::BRUTE_FORCE);//字符匹配算法
	knn->train(TrainDataMat, ROW_SAMPLE, TrainLabelMat);//模型训练

	knn->save("knn_model.xml");

	//预测及结果显示
	double count = 0.0;
	Scalar color;
	for (int i = 0; i < TestData.size(); i++)
	{
		//将测试图片转成CV_32FC1，单行形式
		Mat data = TestData[i].mat.reshape(0, 1);
		data.convertTo(data, CV_32FC1);
		Mat sample = data(Range(0, data.rows), Range::all());

		float f = knn->predict(sample); //预测
		if (f == TestData[i].label)
		{
			color = Scalar(0, 255, 0); //如果预测正确，绘制绿色，并且结果+1
			count++;
		}
		else
		{
			color = Scalar(0, 0, 255);//如果预测错误，绘制红色
		}

		rectangle(src, TestData[i].rect, color, 2);
	}

	//将绘制结果拷贝到一张新图上
	Mat result(Size(src.cols, src.rows + 50), CV_8UC3, Scalar::all(255));
	src.copyTo(result(Rect(0, 0, src.cols, src.rows)));
	//将得分在结果图上显示
	char text[10];
	int score = (count / testNum) * 100;
	sprintf_s(text, "%s%d%s", "Score:", score, "%");
	putText(result, text, Point((result.cols / 2) - 80, result.rows - 15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	imshow("test", result);
	waitKey(0);

	return 0;
}
