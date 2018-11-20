#ifndef __LOAD_PARAM__
#define __LOAD_PARAM__

#include <string>

using namespace std;

class load_param {

public:
	//set window size and step for local shift
	int sizeH = 100; // 200/100
	int stepH = (sizeH / 2); //set windows width and windows step
	
	//set threshold for b/w + limit for x and y shift
	int bw_thr = 70; //75
	int lim_x = 5;  // 0 = no shift on X-Axis
	double lim_y = (sizeH / 4); //50
	//double lim_y =  (sizeH / 2);

	//bool
	bool blob_detection = 1;
	bool crop_images_glob = 1;

	//set paths
	string deposit;
	string path_gen;
	string path_orig;
	string path_affine;
	string pairs;
	string numeric_results;
	string img_results;


public:

	load_param();
	//load_variable();
	void load_path();

};



#endif