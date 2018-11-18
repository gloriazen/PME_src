#ifndef _VARIATIONAL_EROSION_
#define _VARIATIONAL_EROSION_

#include "librerie.h"
#include "load_param.h"


using namespace std;
using namespace cv;


class variational_erosion
{
	cuda::GpuMat img1_g_GPU, img_absdiff_GPU;
	int iter;
	bool save_img;
	string pid_;

public:

	variational_erosion(cuda::GpuMat img1_g_GPU, cuda::GpuMat img_absdiff_GPU, int iter, bool save_img, string pid_);
	vector<double> compute_erosion(load_param& p);
};


#endif