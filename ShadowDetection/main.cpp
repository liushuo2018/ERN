#include "stdafx.h"
#include<opencv2\opencv.hpp>


using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{

	// input the orginal image;
	Mat img = imread("E:\\share\\LH\\VH_ERN\\raw\\top\\top_mosaic_09cm_area11.tif");

	// input the corresponding semantic ground truth
	Mat gt = imread("E:\\share\\LH\\VH_ERN\\raw\\gts_for_participants\\top_mosaic_09cm_area11.tif");

	int Thr = 50;

	int w = img.cols;
	int h = img.rows;

	Mat grey;
	grey = Mat::zeros(cvSize(w, h), CV_8UC1);

	Mat tmp;
	tmp = Mat::zeros(cvSize(w, h), CV_8UC3);

	// contrast preserving decolorization
 	decolor(img, grey, tmp);

	Mat grey2;
	cvtColor(img, grey2, COLOR_BGR2GRAY);

	Mat shadow;
	shadow = Mat::zeros(cvSize(w, h), CV_8UC1);

	int _N = 0;
	int _n = 0;

	// find the shadow regions by a fixed threshold
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			_N = _N + 1;
			if (grey.ptr<uchar>(i)[j] < Thr)
			{
				shadow.ptr<uchar>(i)[j] = 255;
				_n = _n + 1;
			}

			/*
			// Shadow-no-Car
			if (gt.ptr<uchar>(i)[j * 3 + 0] == 0 && gt.ptr<uchar>(i)[j * 3 + 1] == 255 && gt.ptr<uchar>(i)[j * 3 + 2] == 255)
			{
				shadow.ptr<uchar>(i)[j] = 0;
				_n = _n - 1;
			}*/
		}
	}


	// save the results

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	cout << "0" << endl;


	string path;
	path = "E:\\share\\LH\\VH_ERN\\raw\\shadow_png\\top_mosaic_09cm_area11.png";


	imwrite(path, shadow, compression_params);

	float ratio = float(_n) / float(_N);

	cout << ratio << endl;

	return 0;

}
