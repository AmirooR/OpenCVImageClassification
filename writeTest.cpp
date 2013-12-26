#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	FileStorage fs("test.yml", FileStorage::WRITE);

	string name("salam");
	int x[] = {3,6,10};
	fs<<"names"<<"[";
	fs << "{:" << "name" << name.c_str() << "codes" << "[:";
	for(int i = 0; i < 3; i++)
	{
		fs << i << x[i];
	}
	fs << "]" << "}" << "]";
	fs.release();
	return 0;
}
