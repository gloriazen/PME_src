#include "crop_img_glob.h"

using namespace std;
using namespace cv;

crop_img_glob::crop_img_glob(Mat img_, string pid_) : img(img_), pid(pid_) { }

cv::Rect crop_img_glob::crop(load_param& p) {
	Rect rect_crop;
	//	load_param p();
	string sx;
	stringstream s;
	//	s << p.pid_path << pid << ".txt";
	s << p.path_affine << "matrix_" << pid << ".txt";
	sx = s.str();
	ifstream tform_(sx);
	int x, y; double tform[3][3] = { 0 };

	if (!tform_) { cout << "Cannot open file " << sx << "(" << p.path_affine << ").\n"; }
	for (x = 0; x < 3; x++) {
		for (y = 0; y < 3; y++) {
			tform_ >> tform[x][y];
		}
	}	tform_.close();
	/*for (x = 0; x < 3; x++) {
		for (y = 0; y < 3; y++) {
			cout << tform[x][y] << " ";
		} cout << "\n";
	}*/

	int Width, Height, cropX, cropY;
	cv::Size sz = img.size();
	int col = sz.width;
	int row = sz.height;
	//sin crop
	int sincrop_y = abs(sin(tform[0][1]))*col;
	int	sincrop_x = abs(sin(tform[0][1]))*row;

	//sin crop + X-Y shift
	cropX = sincrop_x + abs(tform[0][2]);
	cropY = sincrop_y + abs(tform[1][2]);

	int c = 2;
	Height = row - c * cropY;
	Width = col - c * cropX;

	rect_crop = Rect(cropX, cropY, Width, Height);

	return rect_crop;
}