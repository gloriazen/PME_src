#include "variational_erosion.h"

using namespace std;
using namespace cv;

variational_erosion::variational_erosion(cuda::GpuMat img1_g_GPU, cuda::GpuMat img_absdiff_GPU, int iter, bool save_img, string pid_) :
	img1_g_GPU(img1_g_GPU), img_absdiff_GPU(img_absdiff_GPU), iter(iter), save_img(save_img), pid_(pid_){ }

vector<double> variational_erosion::compute_erosion(load_param& p) {

	vector<double> MSE;

	cuda::GpuMat grad_x, grad_y, I_master_sobel_0, variational_res_GPU, img1_b_GPU;
	int delta = 0, scale = 1, ddepth = CV_8U;
	cuda::threshold(img1_g_GPU, img1_b_GPU, p.bw_thr, 255, THRESH_BINARY);

	// Gradient X - Y
	Ptr<cv::cuda::Filter> filter_x = cuda::createSobelFilter(img1_b_GPU.type(), ddepth, 1, 0, 3, scale, BORDER_DEFAULT);
	filter_x->apply(img1_b_GPU, grad_x);
	Ptr<cv::cuda::Filter> filter_y = cuda::createSobelFilter(img1_b_GPU.type(), ddepth, 0, 1, 3, scale, BORDER_DEFAULT);
	filter_y->apply(img1_b_GPU, grad_y);
	cuda::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, I_master_sobel_0);
	cuda::threshold(I_master_sobel_0, I_master_sobel_0, p.bw_thr, 255, THRESH_BINARY);

	// MSE(0) + OR operation
	Mat variational_res_cpu;
	cuda::subtract(img_absdiff_GPU, I_master_sobel_0, variational_res_GPU);
	cuda::threshold(variational_res_GPU, variational_res_GPU, 0, 255, THRESH_TOZERO);
	if (save_img) {
		variational_res_GPU.download(variational_res_cpu);
		string imgout1 = p.img_results + "/Txy_"  + pid_ + "_iter_" + "0" + "_sobel.png";
		imwrite(imgout1, variational_res_cpu);
	}

	//cuda::norm(variational_res_GPU, NORM_L2);
	cuda::sqr(variational_res_GPU, variational_res_GPU);
	MSE.push_back(cuda::sum(variational_res_GPU)[0] / (variational_res_GPU.size().height*variational_res_GPU.size().width));
	

	// Dilation + MSE(0) + OR operation
	int an = 3;
	Mat element = getStructuringElement(MORPH_RECT, Size(an * 2 + 1, an * 2 + 1), Point(an, an));
	Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, I_master_sobel_0.type(), element);
	for (int i = 1; i < iter; i++) {
		dilateFilter->apply(I_master_sobel_0, I_master_sobel_0);
		cuda::subtract(img_absdiff_GPU, I_master_sobel_0, variational_res_GPU);
		cuda::threshold(variational_res_GPU, variational_res_GPU, 0, 255, THRESH_TOZERO);
		if (save_img) {
			variational_res_GPU.download(variational_res_cpu);
			stringstream ss; ss << i; string i_c = ss.str();
			string imgout2 = p.img_results + "/Txy_" + pid_ + "_iter_" + i_c + "_sobel.png";
			imwrite(imgout2, variational_res_cpu);
		}
		cuda::sqr(variational_res_GPU, variational_res_GPU);
		MSE.push_back(cuda::sum(variational_res_GPU)[0] / (variational_res_GPU.size().height*variational_res_GPU.size().width));
	}

	return MSE;
}