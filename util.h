#ifndef __UTIL_H_
#define __UTIL_H_
//TODO: May I change the fopen to fstream versions?
#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <cassert>

using namespace std;
using namespace cv;

map<string, map<int, float> > readCodes(string filename)
{
	map<string, map<int, float> >  codes_names;
	FileStorage fs;
	fs.open( filename, FileStorage::READ);
	FileNode names = fs["names"];
	FileNodeIterator it = names.begin(), it_end = names.end();
	int idx = 0;
	for(; it != it_end; ++it, ++idx)
	{
		vector<float> list;
		map<int, float> code;
		string name;
		(*it)["name"] >> name;
		(*it)["codes"] >> list;
		
		for(int i = 0; i < list.size(); i+=2)
		{
			code[(int)list[i]] = list[i+1];
		}

		codes_names[name] = code;
	}
	return codes_names;
}

map<string, vector<float> > readCodesVec(string filename)
{
	map<string, vector< float> >  codes_names;
	FileStorage fs;
	fs.open( filename, FileStorage::READ);
	FileNode names = fs["names"];
	FileNodeIterator it = names.begin(), it_end = names.end();
	int idx = 0;
	for(; it != it_end; ++it, ++idx)
	{
		vector<float> list;
		string name;
		(*it)["name"] >> name;
		(*it)["codes"] >> list;
		codes_names[name] = list;
	}
	return codes_names;
}

map<string, int> readLabels(string root, vector<string>& labels_files, vector<int>& labels)
{
	//TODO: ugly 
	assert( labels.size() == labels_files.size() );
	map<string, int> labels_names;
	for(int i = 0; i < labels_files.size(); i++)
	{
		string file_path = root + labels_files[i];
		FILE* fd = fopen( file_path.c_str(), "r");
		char name[32] = {0};
		while( fscanf(fd,"%s",name) > 0 )
		{
			string thisName(name);
			cerr<<thisName<<" "<<labels[i]<<endl;
			labels_names[thisName] = labels[i];
		}
		fclose(fd);
	}

	return labels_names;
}

map<string, int>  getLicensePartialLabels()
{
	string root("/Users/amirrahimi/temp/Gomrok/data/isLicense/");
	vector<string> labels_files;
	vector<int> labels;

	labels_files.push_back("partial0/label0.txt");
	labels.push_back(-1);

	labels_files.push_back("partial1/label1.txt");
	labels.push_back(1);

	return readLabels(root, labels_files, labels);
}

void writeLIBSVMFormat( map<string, int>& names_labels, map<string, vector<float> >& names_codes, string filename)
{
	map< string, vector<float> >::iterator iter;
	FILE* fd = fopen( filename.c_str(), "w");
	for(iter = names_codes.begin(); iter != names_codes.end(); ++iter)
	{
		vector<float> code = iter->second;
		string name = iter->first;

		if( names_labels.find( name ) != names_labels.end())
		{
			int label = names_labels[name];
			fprintf(fd,"%d ",label);
			for(int i = 0; i < code.size(); i+=2)
			{
				fprintf(fd,"%d:%.8f ", (int)code[i], code[i+1]);
			}
			fprintf(fd,"\n");
		}
		else
		{
			cerr<<"Error name "<<name<<" is in code but not in labels!"<<endl;
		}
	}

	fclose(fd);
}


void writePartialLicensesLIBSVMFormat( string filename)
{
	map<string, int>  labels = getLicensePartialLabels();
	map<string, vector<float> > codes = readCodesVec("allCodes.yml"); 
	writeLIBSVMFormat( labels, codes, filename);
}

void writePartialUnlabeledLicensesLIBSVMFormatAndNames(string filename, string names_file)
{
	string root("/Users/amirrahimi/temp/Gomrok/data/isLicense/partial_not_labeled/");
	vector<string> labels_files;
	vector<int> labels;

	labels_files.push_back("imgList.txt");
	labels.push_back(-1);

	map<string,int> labelsMap = readLabels(root, labels_files, labels);
	map<string, vector<float> > codes = readCodesVec("otherCodes.yml"); 
	writeLIBSVMFormat( labelsMap, codes, filename);
	map< string, vector<float> >::iterator iter;

	FILE* fd = fopen(names_file.c_str(), "w");
	for(iter = codes.begin(); iter != codes.end(); ++iter)
	{
		string thisName = iter->first;
		fprintf(fd,"%s\n",thisName.c_str());
	}
	fclose(fd);
}
#endif
