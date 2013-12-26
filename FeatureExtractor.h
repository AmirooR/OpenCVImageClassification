#ifndef __FEATURE_EXTRACTOR_H_
#define __FEATURE_EXTRACTOR_H_

#include <opencv2/opencv.hpp>
#include "drwnHOGFeatures.h"
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace std;
using namespace cv;

typedef enum
{
	HOG_FEATRE_EXTRACTOR,
}FeatureExtractorType;

class FeatureExtractor
{
	protected:
	FeatureExtractorType type;

	public:
	virtual vector< vector<float> > computeFeatures(Mat& img) = 0;
	
	FeatureExtractorType getType()
	{
		return type;
	}
};


class HOGFeatureExtractor: public FeatureExtractor
{
	bool doHistEqualize;
	public:
	HOGFeatureExtractor(bool doHistEqualize = true):FeatureExtractor(),doHistEqualize(doHistEqualize)
	{
		this->type = HOG_FEATRE_EXTRACTOR;
	}
	
	vector< vector<float> > computeFeatures(Mat& img)
	{
		vector< vector<float> > features;
		Mat hog_gray = img.clone();
		if(img.channels() == 3)
		{
			cvtColor(img, hog_gray, CV_BGR2GRAY);
		}

		if ( doHistEqualize )
		{
			equalizeHist( hog_gray, hog_gray);
		}
		Mat hog_im_double(hog_gray);
		hog_gray.convertTo(hog_im_double, CV_32FC1, 1.0/255.0);

		IplImage hog_dimg = hog_im_double;
		drwnHOGFeatures d;
		vector<CvMat*> hFeatuers;
		d.computeFeatures(&hog_dimg, hFeatuers);

		if( hFeatuers.size() > 0 )
		{
			/*for(int f = 0; f < hFeatuers.size(); f++)
			  {
			  vector<float> thisF;
			  cv::Mat tmp( hFeatuers[f] );

			  int numRows = tmp.rows;
			  int numCols = tmp.cols;
			  for(int i = 0; i < numRows; i++)
			  {
			  for(int j = 0; j < numCols; j++)
			  {
			  thisF.push_back(tmp.at<float>(i,j) );
			  }
			  }

			  features.push_back(thisF);
			  }*/


			cv::Mat tmp1( hFeatuers[0] );

			int numRows = tmp1.rows;
			int numCols = tmp1.cols;
			for(int i = 0; i < numRows; i++)
			{
				for(int j = 0; j < numCols; j++)
				{
					vector<float> thisF;

					for(int f = 0; f < hFeatuers.size(); f++)
					{
						cv::Mat tmp( hFeatuers[f] );
						thisF.push_back(tmp.at<float>(i,j) );
					}
					features.push_back(thisF);

				}

			}

		}
		return features;
	}

};

#endif
