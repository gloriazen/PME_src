
#ifndef _CSVREADER_
#define _CSVREADER_

#include "librerie.h"

using namespace std;
using namespace cv;

class CSVReader
{
	std::string fileName;
	std::string delimeter;

public:
	CSVReader(string filename_, string delm = ",");
	vector<vector<string>> getData();
};



#endif
