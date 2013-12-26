#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

Mat rand_select_and_shuffle(Mat in, int num_samples)
{
	if( num_samples > in.rows)
	{
		num_samples = in.rows;//just shuffle
	}

	int total = in.rows;
	int* idx = new int[total];
	for(int i = 0; i < total; i++)
		idx[i] = i;

	srand(time(0));
	random_shuffle(&idx[0], &idx[total-1]);

	Mat retMe;
	retMe.create( num_samples, in.cols, in.type());
	for(int i = 0; i < num_samples; i++)
	{
		in.row( idx[i] ).copyTo( retMe.row(i) );
	}

	return retMe;

}


int run(int clusterCount = 1024, int attemps = 1, float subsampleRatio = 0.125, int numKmeansIteration = 10000, float eps = 0.00001)
{
	cout<<"Reading allHogs.yml ..."<<endl;
	FileStorage fs;
	fs.open("allHogs.yml", FileStorage::READ);
	int num_features;
	fs["NUM_FEATURES"] >> num_features;
	cout<<num_features<<endl;
	cv::Mat allHogs;
	fs["all_hog_features"] >> allHogs;
	cout<<allHogs.cols<<endl;
	cout<<"done!"<<endl;

	Mat labels;
	Mat centers;
	cout<<"Sub-sampling ... "<<endl;
	int subsampleCount = int( subsampleRatio * allHogs.rows);
	Mat subsample = rand_select_and_shuffle( allHogs, subsampleCount);
	cout<<"Running K-Means ... "<<endl;
	kmeans(subsample, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, numKmeansIteration, eps), attemps, KMEANS_PP_CENTERS, centers);
	cout<<"done!"<<endl;
	cout<<"Writing Results ... "<<endl;
	FileStorage fsw("allCenters.yml", FileStorage::WRITE);
	fsw << "centers" << centers;
	fsw.release();
	fs.release();
	return 0;
}


void testRand()
{
	Mat a = Mat::eye(10,10, CV_32S);
	cout<<a<<endl;
	Mat b = rand_select_and_shuffle(a,4);
	cout<<b<<endl;
}

int main()
{
	//testRand();
	run();
	return 0;
}
