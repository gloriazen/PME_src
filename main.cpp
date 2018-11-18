#include "librerie.h"

using namespace std;
using namespace cv;


int main()
{
	//initialize CUDA
	cv::cuda::setDevice(0);	

	// Load parameters and pid images
	load_param p;
	
	int pid_from, pid_to; 
	pid_from =  37;	//21 - 29 - 37 - 47
	pid_to =	42;	//28 - 36 - 46 - 60
	//pid_to = pid_from;

	for (int pid_n = pid_from; pid_n <= pid_to; pid_n++) {

		std::clock_t start; // start clock
		double shift_time;
		//get pid
		ostringstream str1;
		str1 << pid_n;
		string pid_ = str1.str();
		string str_tmp = p.pairs;
		load_pid::imgs im_pair = load_pid(str_tmp, pid_).get_imgs();

		//open time txt
		string Txy_file_time_string = p.numeric_results + "/Txy" + pid_ + "_time.txt";
		ofstream Txy_file; Txy_file.open(Txy_file_time_string);
		Txy_file << "Window Size: " << p.sizeH << "\tImages Pid: " << pid_ << endl << endl;

		// Read the images to be aligned		
		Mat im1 = imread(im_pair.fix);
		Mat im2 = imread(im_pair.mov);
		Mat mask1 = imread(im_pair.mask_fix);
		Mat mask2 = imread(im_pair.mask_mov);
		// Check if img1 and img2 exist
		if (empty(im1) || empty(im2)) {
			cout << "Img 1 or Img2 does not exist";
			return -2;
		}

		//Convert to grayscale, Control size images
		start = std::clock();
		cout << endl << "Convert to grayscale, Control size images: " << endl;

		// Control images sizes
		Size sz1 = im1.size(), sz2 = im2.size();
		int row_g, col_g;
		if (sz1.width < sz2.width) { row_g = sz1.width; }
		else { row_g = sz2.width; }
		if (sz1.height < sz2.height) { col_g = sz1.height; }
		else { col_g = sz2.height; }
		Rect c1 = Rect(0, 0, row_g, col_g);
		im1 = im1(c1);
		im2 = im2(c1);
		mask1 = mask1(c1);
		mask2 = mask2(c1);

		// Convert images to gray scale;
		Mat im1_gray, im2_gray;
		if (im1.channels() == 3) { cvtColor(im1, im1_gray, CV_RGB2GRAY); }
		else { im1_gray = im1; }
		if (im2.channels() == 3) { cvtColor(im2, im2_gray, CV_RGB2GRAY); }
		else { im2_gray = im2; }
		if (mask1.channels() == 3) { cvtColor(mask1, mask1, CV_RGB2GRAY); }
		else { mask1 = mask1; }
		if (mask2.channels() == 3) { cvtColor(mask2, mask2, CV_RGB2GRAY); }
		else { mask2 = mask2; }

		shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Time: " << shift_time << endl;
		Txy_file << "Conv to gray, Control im size:\t" << shift_time << endl;

		// crop imgs after global match, avoiding black side bands		
		Mat img1_g, img2_g;
		if (p.crop_images_glob) {
			Rect crop_images;
			start = std::clock();
			cout << endl << "Crop images: \n";
			crop_images = crop_img_glob(im1_gray, pid_).crop(p);
			img1_g = im1_gray(crop_images); //il crop non avviene correttamente, ricontrollare nuovamente valori in crop_img_glob
			img2_g = im2_gray(crop_images);
			mask1 = mask1(crop_images);
			mask2 = mask2(crop_images);
			shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			std::cout << "Time: " << shift_time << endl;
			Txy_file << "Crop images:\t\t\t" << shift_time << endl;
		}
		else {
			img1_g = im1_gray;
			img2_g = im2_gray;
		}
		//namedWindow("Image 1", CV_WINDOW_NORMAL); imshow("Image 1", img1_g); cv::waitKey();
		//namedWindow("Image 2", CV_WINDOW_NORMAL); imshow("Image 2", img2_g); cv::waitKey();


		//GET LOCAL SHIFT//
		start = std::clock();
		cout << endl << "Finding local shift: " << endl;
		local_shift_calc::Txy Txy_ = local_shift_calc(img1_g, img2_g, mask1, mask2).get_shift(p);
		shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Time: " << shift_time << endl;
		Txy_file << "Calculate shift:\t\t" << shift_time << endl;
		//end of local shift calc

		//INTERPOLATION//
		start = std::clock();
		cout << endl << "X - Y interpolation:" << endl;

		//spacing sample
		int ROW_N = 0;
		int bound_row = img2_g.size().height - (p.stepH * 2);
		for (int i = 0; i < bound_row; i = i + p.stepH) { ROW_N++; }

		std::vector<double> step, X, Y; // X->stepH:stepH:ROW for Y values 
		for (int i = 0; i < ROW_N; i++) { //eventualmente qui considerare limitazioni con std, lim_y ecc
			if (!isnan(Txy_.Ty.at(i))) {
				step.push_back((i + 1) * p.stepH);
				X.push_back(Txy_.Tx.at(i));
				Y.push_back(Txy_.Ty.at(i));
			}
			else {
				step.push_back((i + 1) * p.stepH);
				X.push_back(0);
				Y.push_back(0);
			}
		}

		//interpolation function
		tk::spline s_x, s_y;
		s_x.set_points(step, X);
		s_y.set_points(step, Y);

		//get and save X-Y pixel shift
		Mat interp_x_vec = Mat(cv::Size(1, img1_g.size().height), CV_32FC1);
		Mat interp_y_vec = Mat(cv::Size(1, img1_g.size().height), CV_32FC1);
		for (int i = 0; i < img1_g.size().height; i++) {
			interp_y_vec.at<float>(i) = s_y(i) + i;
		}
		for (int i = 0; i < img1_g.size().height; i++) { //get X shift (row index)
			interp_x_vec.at<float>(i) = s_x(i);
		}
		Mat interp_x_vec2 = Mat(cv::Size(1, img1_g.size().width), CV_32FC1);
		for (int i = 0; i < img1_g.size().width; i++) { //get X original position (col index)
			interp_x_vec2.at<float>(i) = i;
		}
		shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Time: " << shift_time << endl;
		Txy_file << "Get Interp values:\t\t" << shift_time << endl;

		//generate Xmap and Ymap in order to remap the moving image
		start = std::clock();
		cout << endl << "Generation of Xmap and Ymap:" << endl;
		Mat Xmap = Mat(img2_g.size(), CV_32FC1);
		Mat Xmap2 = Mat(img2_g.size(), CV_32FC1);
		Mat Ymap = Mat(img2_g.size(), CV_32FC1);

		Xmap = repeat(interp_x_vec, 1, img1_g.size().width);
		Xmap2 = repeat(interp_x_vec2, 1, img1_g.size().height);
		transpose(Xmap2, Xmap2);
		Xmap = Xmap + Xmap2; //X original position + X shift
		Ymap = repeat(interp_y_vec, 1, img1_g.size().width);
		shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Time: " << shift_time << endl;
		Txy_file << "generate Xmap and Ymap:\t\t" << shift_time << endl;


		//remap
		start = std::clock();
		cout << endl << "Remap:" << endl;
		cuda::GpuMat xmap_gpu, ymap_gpu, img_2_g_REMAP_gpu, img1_g_GPU, img_absdiff_GPU_PME, img2_g_GPU;


		//load images on GPU
		xmap_gpu.upload(Xmap);
		ymap_gpu.upload(Ymap);
		img1_g_GPU.upload(img1_g);
		img2_g_GPU.upload(img2_g);

		//GPU remap
		cuda::remap(img2_g_GPU, img_2_g_REMAP_gpu, xmap_gpu, ymap_gpu, INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));
		cuda::absdiff(img_2_g_REMAP_gpu, img1_g_GPU, img_absdiff_GPU_PME);

		shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Time: " << shift_time << endl;
		Txy_file << "Remap image:\t\t\t" << shift_time << endl;
		//end of interpolation


		//DEFECTS LOCALIZATION//
		cuda::GpuMat img_absdiff_GPU_NO_PME;
		cuda::absdiff(img1_g_GPU, img2_g_GPU, img_absdiff_GPU_NO_PME);
		int n_iter = 10;

		vector<double>::iterator max_x, min_x;
		max_x = max_element(X.begin(), X.end());
		min_x = min_element(X.begin(), X.end());
		float max_value_x = *max_x;
		float min_value_x = *min_x;
		if (min_value_x > 0) { min_value_x = 0; }
		if (max_value_x < 0) { max_value_x = 0; }
		cout << "\nMin-Max crop X values:\n(" << min_value_x << ") - (" << max_value_x << ")\n";


		Rect crop(max_value_x, 0, img1_g_GPU.size().width + min_value_x - max_value_x, img1_g_GPU.size().height);
		cuda::GpuMat img1_g_GPU_crop = img1_g_GPU(crop);
		cuda::GpuMat img_absdiff_GPU_PME_crop = img_absdiff_GPU_PME(crop);
		cuda::GpuMat img_absdiff_GPU_NO_PME_crop = img_absdiff_GPU_NO_PME(crop);


		//Variational Erosion
		string img_erosion_string = p.numeric_results + "/Txy" + pid_ + "_erosion.txt";
		ofstream img_erosion; img_erosion.open(img_erosion_string);
		img_erosion << "MSE PME: " << "\tMSE NO PME: " << endl;
		cout << "\nVariational erosion:" << endl;
		start = std::clock();
		vector<double> MSE_PME = variational_erosion(img1_g_GPU_crop, img_absdiff_GPU_PME_crop, n_iter, false, pid_).compute_erosion(p);
		vector<double> MSE_NO_PME = variational_erosion(img1_g_GPU_crop, img_absdiff_GPU_NO_PME_crop, n_iter, false, pid_).compute_erosion(p);
		for (int i = 0; i < n_iter; i++) {
			cout << "Iter " << i << ":\t" << "MSE_PME: " << MSE_PME.at(i) << "\t\tMSE_NO_PME: " << MSE_NO_PME.at(i) << endl;
			img_erosion << MSE_PME.at(i) << "\t" << MSE_NO_PME.at(i) << endl;
		}
		img_erosion.close();
		shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "Time: " << shift_time << endl;
		Txy_file << "Variational erosion:\t\t" << shift_time << endl;


		//ROI//
		Mat absdiff_cpu;
		img_absdiff_GPU_PME_crop.download(absdiff_cpu);
		Mat img_ROI = absdiff_cpu;

		if (p.blob_detection) {

			//load and draw defect ROI// 
			cout << "\nLoad and draw defect ROI\n" << endl;
			start = std::clock();

			//defect dir
			char *path_buffer, drive[_MAX_DRIVE], dir[_MAX_DIR], fname[_MAX_FNAME], ext[_MAX_EXT];
			path_buffer = &im_pair.fix[0u];
			_splitpath_s(path_buffer, drive, dir, fname, ext); // C4996  	
			//defect name
			char *path_buffer_, drive_[_MAX_DRIVE], dir_[_MAX_DIR], fname_[_MAX_FNAME], ext_[_MAX_EXT];
			path_buffer_ = &im_pair.defect[0u];
			_splitpath_s(path_buffer_, drive_, dir_, fname_, ext_); // C4996  	
			//create path defect
			char* _defect = strcat(drive, dir);
			string dir_defect(_defect); dir_defect = dir_defect + "defect_roi.txt";
			cout << dir_defect << endl;
			//create img name
			string img_def_name(fname_);  img_def_name = img_def_name + ext;
			cout << img_def_name << endl;
			//create img name edit
			string img_defect_edit(fname_);
			img_defect_edit = img_defect_edit + "_edit" + ext;
			cout << img_defect_edit << endl << endl;

			//#img	#defect_class	#bbox.min.x	#bbox.min.y #dX #dy
			//1-Cam3_0_edit.png macnza_mezzo_puntino 1083 5580 48 49
			vector<int> ROI_X_v, ROI_Y_v, ROI_dX_v, ROI_dY_v;
			std::ifstream infile(dir_defect);
			if (infile.is_open()) {
				int cont = 0;
				infile.ignore(500, '\n'); //skip first row
				string name, defect_name, check_img;
				int ROI_X, ROI_Y, ROI_dX, ROI_dY;
				while (infile.good()) { //take only the ROI data about the img
					infile >> name >> defect_name >> ROI_X >> ROI_Y >> ROI_dX >> ROI_dY; //put img ROI in a vector
					cout << name << endl;
					if (name == img_def_name || name == img_defect_edit) {	
						ROI_X_v.push_back(ROI_X); ROI_dX_v.push_back(ROI_dX);
						ROI_Y_v.push_back(ROI_Y); ROI_dY_v.push_back(ROI_dY);
						cont++;
					}
				}
			}

			//add ROI to img_ROI
			for (int i = 0; i < ROI_X_v.size(); i++) {
				Rect ROI = Rect(ROI_X_v.at(i), ROI_Y_v.at(i), ROI_dX_v.at(i), ROI_dY_v.at(i));
				cv::rectangle(img_ROI, ROI, Scalar(0, 255, 0), 1, LINE_8, 0);
			}
			shift_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			std::cout << "\nTime: " << shift_time << endl;
		}

		//save images//
		cout << endl << "Saving images:" << endl;

		Mat img_2_g_REMAP_cpu;	img_2_g_REMAP_gpu.download(img_2_g_REMAP_cpu);
		string imgout1 = p.img_results + "/Txy_" + pid_ + "_interp_img.png"; imwrite(imgout1, img_2_g_REMAP_cpu);

		string imgout2 = p.img_results + "/Txy_" + pid_ + "img_diff_PME.png"; imwrite(imgout2, img_ROI); //absdiff_cpu

		Mat img_diff;  absdiff(img1_g, img2_g, img_diff);
		string imgout4 = p.img_results + "/Txy_" + pid_ + "img_diff_original.png"; imwrite(imgout4, img_diff);




		Txy_file.close();
		std::cout << "\n\n \t\t Bye bye, World!" << std::endl;
	}
	return 0;
}


