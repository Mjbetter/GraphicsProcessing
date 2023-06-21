#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
#include <qdebug.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

char ad[14059010] = { 0 };

int main()
{
	Mat img = imread("C:\\Users\\MJ\\Desktop\\计算机视觉\\GraphicsProcessing\\data\\digits.png");
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;   //每一行图片的个数
	int n = gray.cols / b;   //每一列图片的个数
	Mat data, labels;   //特征矩阵
	for (int i = 0; i < n; i++)
	{
		int offsetCol = i * b; //列上的偏移量
		for (int j = 0; j < m; j++)
		{
			int offsetRow = j * b;  //行上的偏移量
									//截取20*20的小块
			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			data.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵
			labels.push_back((int)j / 5);  //对应的标注
		}

	}

	Mat gray_frame, thres_img, blur_img;
	Mat morph_img, tmp2, tmp3;
	Mat kernerl = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	for (int k = 0; k < 10; ++k) {
		//加入自己写的数据集
		memset(ad, '0', 14059010);
		sprintf_s(ad, "C:\\Users\\MJ\\Desktop\\计算机视觉\\GraphicsProcessing\\data\\%d.jpg", k);
		Mat src = imread(ad);

		cvtColor(src, gray_frame, COLOR_BGR2GRAY);//对图像进行预处理（图像去燥二值化）
		GaussianBlur(gray_frame, blur_img, Size(3, 3), 3, 3);
		adaptiveThreshold(blur_img, thres_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 151, 10);
		morphologyEx(thres_img, morph_img, MORPH_OPEN, kernerl, Point(-1, -1));
		//k是显示想要的轮廓数这样就可以方便把新数据压入数据集 不用单独保存
		vector<vector<Point>> contours;
		vector<Vec4i> hiearachy;
		findContours(morph_img, contours, hiearachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); ++i)
		{

			Rect minrect = boundingRect(contours[i]);
			float area = contourArea(contours[i]);
			float ckbi = minrect.width / minrect.height;
			//cout << ckbi << endl;
			if (ckbi < 4 && area>50)
			{
				//cout << minrect.height << endl << minrect.width;
				rectangle(src, minrect, Scalar(0, 255, 0), 1, 8);
				Rect ROI = minrect;
				Mat ROI_img = morph_img(ROI);
				resize(ROI_img, ROI_img, Size(20, 20));
				ROI_img.copyTo(tmp2);
				stringstream stream;
				stream << k;
				string str;
				stream >> str;
				//string_temp = stream.str();
				putText(src, str, ROI.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 1, 8);
				data.push_back(tmp2.reshape(0, 1));  //序列化后放入特征矩阵
				labels.push_back(k);  //对应的标注
			}
		}
	}



	data.convertTo(data, CV_32F); //uchar型转换为cv_32f
	int samplesNum = data.rows;
	int trainNum = 5000;
	Mat trainData, trainLabels;
	trainData = data(Range(0, samplesNum), Range::all());   //前3000个样本为训练数据
	trainLabels = labels(Range(0, samplesNum), Range::all());//前三千个训练标签
	//使用KNN算法
	int K = 7;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);
	model->save("KnnTest.xml");

	//用样本进行测试
	Mat src_test = imread("C:\\Users\\MJ\\Desktop\\testImg\\手写数字.jpg");
	Mat gray_test, thres_test, blur_test;
	cvtColor(src_test, gray_test, COLOR_BGR2GRAY);
	GaussianBlur(gray_test, blur_test, Size(3, 3), 3, 3);
	adaptiveThreshold(blur_test, thres_test, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 151, 10);
	Mat morph_test, predict_mat;
	morphologyEx(thres_test, morph_test, MORPH_OPEN, kernerl, Point(-1, -1));
	vector<vector<Point>> contours_test;
	vector<Vec4i> hiearachy_test;
	findContours(morph_test, contours_test, hiearachy_test, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int count = 0;
	for (int i = 0; i < contours_test.size(); ++i)
	{

		Rect minrect_test = boundingRect(contours_test[i]);
		float area_test = contourArea(contours_test[i]);
		float ckbi_test = minrect_test.width / minrect_test.height;
		if (ckbi_test < 4 && area_test>50)
		{
			rectangle(src_test, minrect_test, Scalar(0, 255, 0), 1, 8);
			Rect ROI_test = minrect_test;
			Mat ROI_img_test = morph_test(ROI_test);
			resize(ROI_img_test, ROI_img_test, Size(20, 20));
			ROI_img_test.convertTo(ROI_img_test, CV_32F);
			ROI_img_test.copyTo(tmp3);
			predict_mat.push_back(tmp3.reshape(0, 1));
			count++;
			Mat predict_simple = predict_mat.row(count-1);
			float r = model->predict(predict_simple);
			stringstream stream;
			stream << r;
			string str;
			stream >> str;
			putText(src_test, str, ROI_test.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 1, 8);
		}

	}
	
	return 0;
}
