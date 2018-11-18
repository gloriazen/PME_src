#ifndef _CROP_IMG_GLOB_
#define _CROP_IMG_GLOB_

#include "librerie.h"
#include "load_param.h"

using namespace std;
using namespace cv;

class crop_img_glob
{
	Mat img;
	string pid;

public:
	crop_img_glob(Mat img_, string pid_);
	Rect crop(load_param& p);
};

#endif