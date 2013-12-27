//TODO: save index
//TODO: do I need to normalize each codes using the computed distances?
//TODO: do I need to normalize the final pooled codes?
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "FeatureExtractor.h"
#include "DataSetReader.h"
#include <map>
#include <set>

using namespace std;
using namespace cv;
using namespace flann;
using namespace cvflann;

flann::Index trainKdtree(string centers_path, int n_kdd = 4)
{
	cout<<"Reading Centers ...";
	FileStorage fs;
	fs.open(centers_path, FileStorage::READ);
	Mat centers;
	fs["centers"]>>centers;
	cout<<centers.rows<< ", "<<centers.cols<<endl;
	cout<<"\t\t\tdone!"<<endl;
	fs.release();
	flann::KDTreeIndexParams params(n_kdd);
	cout<<"Training ...";
	flann::Index index(centers, params);
	cout<<"\t\t\tdone!"<<endl;
	return index;
}

map<int,float> pool( vector<vector<float> >& codes)
{
	map< int, float> indexToPooledDistanceMap;
	for(int i = 0; i < codes.size(); i++)
	{
		for(int j =0; j < codes[i].size(); j+=2)
		{
			if( indexToPooledDistanceMap.find((int)codes[i][j]) != indexToPooledDistanceMap.end() )
			{
				indexToPooledDistanceMap[(int)codes[i][j]] = max( indexToPooledDistanceMap[(int)codes[i][j]], codes[i][j+1]);
			}
			else
			{
				indexToPooledDistanceMap[(int)codes[i][j]] = codes[i][j+1];
			}
		}
	}
	return indexToPooledDistanceMap;
}

map<int,float>  code(flann::Index& index, vector<vector<float> >& features, int knn = 5)
{
	
	vector< vector<float> > codes;

	for(int i = 0; i < features.size(); i++)
	{
		vector<float> dists;
		vector<int> indices;
		vector<float> data = features[i];
		
		index.knnSearch(data, indices,  dists, knn);
		vector<float> thisCodes;
		for(int j =0; j < knn; j++)
		{
			thisCodes.push_back( indices[j] );
			thisCodes.push_back( dists[j] );
		}

		codes.push_back(thisCodes);

	}


	return pool(codes);
}

void writeAllCodes(vector<string>& names, vector< map<int, float> >& codes, string filename )
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs<<"names"<<"[";
	for( int i = 0; i < names.size(); i++)
	{
		fs<< "{:"<<"name"<<names[i].c_str() << "codes" << "[:";
		map<int, float>::iterator iter;
		for(iter = codes[i].begin(); iter != codes[i].end(); ++iter)
		{
			fs << iter->first << iter->second;
		}
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();
}

int main()
{
	
	int knn = 5;
	int n_kdd = 1;
	string centers_path("allCenters.yml");

	/*use this for partial training
	  DataSetReader* dsReader = new DirReader("/Users/amirrahimi/temp/Gomrok/data/isLicense/partialAll/",
			"imgList.txt", 
			2926);*/

	DataSetReader* dsReader = new DirReader("/Users/amirrahimi/temp/Gomrok/data/isLicense/partial_not_labeled/",
			"imgList.txt", 
			4223);
	FeatureExtractor* fextractor = new HOGFeatureExtractor();

//	flann::Index index = trainKdtree("allCenters.yml", n_kdd);
	cout<<"Reading Centers ...";
	FileStorage fs;
	fs.open(centers_path, FileStorage::READ);
	Mat centers;
	fs["centers"]>>centers;
	cout<<centers.rows<< ", "<<centers.cols<<endl;
	cout<<"\t\t\tdone!"<<endl;
	fs.release();
	flann::KDTreeIndexParams params(n_kdd);
	cout<<"Training ...";
	flann::Index index(centers, params);
	cout<<"\t\t\tdone!"<<endl;

	cout<<"Coding ..."<<endl;
	int idx = 0;

	vector<string> allNames;
	vector< map<int, float> > allCodes;
	while( dsReader -> hasNext() )
	{
		Mat img = dsReader->getNext();
		string name = dsReader->getThisName();
		idx++;
		cerr<<"["<<idx<<"] "<< name << "  ... ";
		if( img.rows > 16 && img.cols > 80 )
		{
			vector< vector<float> >  tFeautes = fextractor->computeFeatures(img);
			cerr<<"Features Extracted ...";
			map<int,float>  codes = code(index, tFeautes, knn);
			cerr<<"Coding done!"<<endl;
			allNames.push_back( name);
			allCodes.push_back( codes);
		}
	}

	cout<< "\t\t\tdone!"<<endl;
	cout<< "Writing Results ... ";

	// use this for partial training writeAllCodes( allNames, allCodes, "allCodes.yml");

	writeAllCodes( allNames, allCodes, "otherCodes.yml");

	cout<< "\t\t\tdone!"<<endl;
	delete dsReader;
	delete fextractor;
	return 0;
}
