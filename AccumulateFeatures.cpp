#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "drwnHOGFeatures.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include "FeatureExtractor.h"
#include "DataSetReader.h"


using namespace cv;
using namespace std;


int main()
{
	DataSetReader* dsReader = new DirReader("/Users/amirrahimi/temp/Gomrok/data/isLicense/partialAll/",
			"imgList.txt", 
			//2);
			2926);

	vector<vector<float> > features;
	FeatureExtractor* fextractor =new HOGFeatureExtractor();
	FileStorage fs("allHogs.yml", FileStorage::WRITE);
	while( dsReader-> hasNext() )
	{
		Mat img = dsReader->getNext();
		if( img.rows > 16 && img.cols > 80 )
		{
			vector<vector<float> > tFeautes = fextractor->computeFeatures(img);
			features.insert( features.end(), tFeautes.begin(), tFeautes.end() );
		}
	}
	cout<<"NUM Features: "<< features.size() <<endl;
	stringstream ss;
	int i2 = features.size();
	ss>>i2;
	cout<<i2<<endl;
	fs<< "NUM_FEATURES" << i2;
	cv::Mat allHOGs( features.size(), features.at(0).size(), CV_32FC1);
	for(int i = 0; i < allHOGs.rows; i++)
		for( int j = 0; j < allHOGs.cols; j++)
			allHOGs.at<float>(i, j) = features.at(i).at(j);
	fs<< "all_hog_features"<< allHOGs;
	fs.release();
	delete fextractor;
	delete dsReader;
	return 0;
}
