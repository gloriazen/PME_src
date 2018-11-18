#include "local_shift_calc.h"

using namespace std;
using namespace cv;

local_shift_calc::local_shift_calc(Mat img1, Mat img2, Mat mask_fix, Mat mask_mov) : img1(img1) , img2(img2), mask_mov(mask_mov), mask_fix(mask_fix){ }

local_shift_calc::Txy local_shift_calc::get_shift(load_param& p){
	
	Txy Txy_;
	vector<float> Tx, Ty, Tt;
	//Mat img1_crop, img2_crop, img1_crop_borders;
	cv::Size yEnd_sz,sz = img2.size();	int row = sz.width-1, col = sz.height-1;
	vector<int> yEnd;
	for (int i = 0; i < col-(p.stepH*2); i = i + p.stepH ) { yEnd.push_back(i);	}
	int ROW_N = yEnd.size() - 1; 
	vector<int> dec_x, dec_y; 
	int left = p.lim_x; int right = left;

	//Binary image
	cuda::GpuMat img1_b_GPU, img2_b_GPU, mask_fix_b_GPU, mask_mov_b_GPU;

	cuda::threshold(img1, img1_b_GPU, p.bw_thr, 255, THRESH_BINARY);
	cuda::threshold(img2, img2_b_GPU, p.bw_thr, 255, THRESH_BINARY);
	if (!empty(mask_fix)) {
		cuda::threshold(mask_fix, mask_fix_b_GPU, p.bw_thr, 1, THRESH_BINARY);
		cuda::multiply(img1_b_GPU, mask_fix_b_GPU, img1_b_GPU);
	}
	if (!empty(mask_mov)) {
		cuda::threshold(mask_mov, mask_mov_b_GPU, p.bw_thr, 1, THRESH_BINARY);
		cuda::multiply(img2_b_GPU, mask_mov_b_GPU, img2_b_GPU);
	}

	cuda::GpuMat result, image_d_border;
	double max_value;
	cv::Point location;
	//Mat BW; img2_b_GPU.download(BW);namedWindow("Image bw", CV_WINDOW_NORMAL); imshow("Image bw", BW); cv::waitKey();

	cuda::GpuMat grad_x, grad_y;
	int delta = 0, scale = 1, ddepth = CV_8U;
	// Gradient X
	Ptr<cv::cuda::Filter> filter_x = cuda::createSobelFilter(img2_b_GPU.type(), ddepth, 1, 0, 3, scale, BORDER_DEFAULT);
	filter_x->apply(img2_b_GPU, grad_x);
	// Gradient X
	Ptr<cv::cuda::Filter> filter_y = cuda::createSobelFilter(img2_b_GPU.type(), ddepth, 0, 1, 3, scale, BORDER_DEFAULT);
	filter_y->apply(img2_b_GPU, grad_y);


	for (int i = 0; i <= ROW_N; i++) {

		Rect croppedImage2 = Rect(0, yEnd[i], row, p.sizeH);		
		cuda::GpuMat templ_d = img2_b_GPU(croppedImage2);

		gradient_criteria criteria2;		
		double s_x = cuda::sum(grad_y(croppedImage2))[0];
		double s_y = cuda::sum(grad_x(croppedImage2))[0];
		if (s_y < 100 && s_x < 100) {		//soglia spannometrica
			criteria2.dec_x = 0;	criteria2.dec_y = 0;	criteria2.NTHR = 0;
		}
		else {
			criteria2.dec_x = 1;	criteria2.dec_y = 1;	criteria2.NTHR = 1;
		}

		if ((criteria2.dec_x > 0 || criteria2.dec_y > 0) && i>1 && i<ROW_N) {
			
			Rect croppedImage1 = Rect(0, yEnd[i]-p.lim_y, row, p.sizeH + (p.lim_y * 2)); //img1_crop with size_y > img2_crop
			cuda::GpuMat image_d = img1_b_GPU(croppedImage1);			
			cuda::copyMakeBorder(image_d, image_d_border, 0, 0, left, right, BORDER_CONSTANT, 255); //add black/white borders on left and right sides
			
			//cuda match template				
			cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(templ_d.type(), CV_TM_CCOEFF_NORMED);  //CV_TM_CCOEFF_NORMED // CV_TM_CCORR		
			alg->match(image_d_border, templ_d, result);
			cv::cuda::minMaxLoc(result, 0, &max_value, 0, &location);

			int y_ = (location.y - p.lim_y)*(-1); //delete constant Y shift and invert movement (location.y - p.stepH)*(-1);
			Ty.push_back(y_);
			int x_ = (location.x - left)*(-1); //delete constant X shift
			Tx.push_back(x_); 

			if ((criteria2.dec_y) > 0){ //0.6*criteria2.NTHR) { //da eliminare
				Tt.push_back(2);
			}
			else { Tt.push_back(0); }
		}
		else {
			Tx.push_back(NAN);
			Ty.push_back(NAN);
			Tt.push_back(0);
		}
		Txy_.Tx = Tx;
		Txy_.Ty = Ty;
		Txy_.Tt = Tt;
	}

	return Txy_;
}
