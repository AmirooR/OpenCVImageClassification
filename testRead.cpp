#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int test1()
{
	FileStorage fs;
	fs.open("allHogs.yml", FileStorage::READ);
	int num_features;
	fs["NUM_FEATURES"] >> num_features;
	cout<<num_features<<endl;
	cv::Mat allHogs;
	fs["all_hog_features"] >> allHogs;
	cout<<allHogs.cols<<endl;
	return 0;
}


void test2()
{
	FileStorage fs;
	fs.open("test.yml", FileStorage::READ);
	FileNode names = fs["names"];
	FileNodeIterator it = names.begin(), it_end = names.end();
	int idx = 0;
	vector<float> codes;	
	for(; it != it_end; ++it, ++idx)
	{
		string n;
		(*it)["name"] >> n;
		cout<<"["<<idx<<"] "<< n;
		(*it)["codes"] >> codes;
		//cout<<"\t"<<codes<<endl;
		cout<<"\t";
		for(int i = 0; i < codes.size(); i++)
		{
			cout<<codes[i]<<", ";
		}
		cout<<endl;
	}

}

int main()
{
	test2();
	vector<float> ii;
	
	for(int i = 0;i < 10; i++)
	{
		ii.push_back(i+0.5f);
	}

	Mat m(1,ii.size(), CV_32FC1, &ii[0]);
	cout<<m<<endl;
	for(int i =0; i <2000 ; i++)
	{
		float x = (float)i;
		int y = (int)x;
		if( i != y )
		{
			cout<<"Error: float = "<< x << ", int = "<< i << " int(float) = " << y <<endl;
		}
	}
	return 0;
}
