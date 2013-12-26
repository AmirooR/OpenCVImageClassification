#ifndef __DATA_SET_READER_H_
#define __DATA_SET_READER_H_
//TODO: DirReader: cases where path is wrong and fd is invalid
//TODO: DirReader: remove the need of getting total

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class DataSetReader
{
	public:
		virtual bool hasNext() = 0;

		virtual Mat getNext() = 0;

		virtual string getThisName() = 0;

		virtual ~DataSetReader(){};
};

class DirReader: public DataSetReader
{
	string root;
	string listFile;
	string thisName;
	int total;
	int numRead;
	FILE* fd;
public:
	DirReader(string root, string listFile,int total): DataSetReader(), 
		root(root),
		listFile(listFile),
		total(total),
		numRead(0),
		thisName("")
	{
		string imgListFile = root + listFile;
		fd = fopen( imgListFile.c_str(), "r");
		if( fd <= 0 )
			cout<<"Error opening filelist"<<endl;
	}

	bool hasNext()
	{
		if( numRead < total)
		{
			return true;
		}
		return false;
	}

	Mat getNext()
	{
		char name[32] = {0};
		fscanf(fd, "%s", name);
		thisName = name;
		string imgName = root;
		imgName += name;
		Mat img = imread(imgName.c_str());
		numRead++;
		return img;
	}

	string getThisName()
	{
		return thisName;
	}

	~DirReader()
	{
		if(fd > 0)
			fclose(fd);
		fd = 0;
	}

};

#endif
