#include "load_pid.h"

using namespace std;
using namespace cv;

load_pid::load_pid(string file_, string pid_) : file(file_), pid(pid_) {}

load_pid::imgs load_pid::get_imgs(){

	imgs img;
	load_param p;
	// Creating an object of CSVWriter
	CSVReader reader(p.pairs);
	// Get the data from CSV File
	vector<vector<string> > dataList = reader.getData();
	// Print the content of row by row on screen
	int i = 0, ii = 0; 
	string fix_;
	string defect_;
	for (vector<string> vec : dataList) {
		for (string data : vec) {
			cout << data << ",";
		}
		cout << std::endl;
		if ((dataList[i][1] == pid) && (dataList[i][4] == "1")) {
			fix_= dataList[i][3];
		}
		if ((dataList[i][1] == pid) && (dataList[i][4] == "2"))  {
			defect_ = dataList[i][3];
		}
		i++;
	}
	cout << "\n\pid: " << pid;
	cout << "\nFixed Image:\t" << fix_;
	cout << "\nMoving Image:\t" << "affine_inv_" << pid << ".png\n";
	cout << "\nWindow Size: " << p.sizeH << " Window Step: " << p.stepH;
	cout << "\nLim x: " << p.lim_x << " Lim y: " << p.lim_y << endl;

	string fix = p.path_orig + fix_;
	string mov = p.path_affine + "affine_inv_" + pid + ".png"; //affine_inv_
	string mask_mov = p.path_affine + "glass_region_" + pid + ".png";
	//string mask_fix = p.path_affine + "glass_region_" + pid + ".png";
	string defect = p.path_orig + defect_;

	img.fix = fix;
	img.mov = mov; 
	img.mask_mov = mask_mov; 
	img.mask_fix = mask_mov; //mask_fix
	img.defect = defect;
	return img;
};



