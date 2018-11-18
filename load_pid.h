#ifndef _LOADPID_
#define _LOADPID_

#include "librerie.h"

using namespace std;
using namespace cv;

class load_pid
{
	string file;
	string pid;
	
public:
	struct imgs { string mov, fix, mask_mov, mask_fix, defect; };
	load_pid(string file_, string pid_);
	imgs get_imgs();
};



#endif
