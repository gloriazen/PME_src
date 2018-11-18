#ifndef __LOCAL_SHIFT_CALC__
#define __LOCAL_SHIFT_CALC__

#include "librerie.h"
#include "load_param.h"

using namespace std;
using namespace cv;

class local_shift_calc
{	
	Mat img1, img2, mask_mov, mask_fix;
	struct gradient_criteria { int dec_x, dec_y, NTHR; };

public:
	struct Txy { vector<float> Tx, Ty, Tt; };
	local_shift_calc(Mat img1, Mat img2, Mat mask_fix, Mat mask_mov);
	local_shift_calc::Txy get_shift(load_param& p);
};

#endif