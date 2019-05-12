#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;

static double rad2Deg(double rad) {
	return rad * (180 / M_PI);
} //Convert radians to degrees

static double deg2Rad(double deg) {
	return deg * (M_PI / 180);
} //Convert degrees to radians

void warpMatrix(Size sz, double theta, double phi, double gamma, double scale,
		double fovy, Mat& M, vector<Point2f>* corners) {
	double st = sin(deg2Rad(theta));
	double ct = cos(deg2Rad(theta));
	double sp = sin(deg2Rad(phi));
	double cp = cos(deg2Rad(phi));
	double sg = sin(deg2Rad(gamma));
	double cg = cos(deg2Rad(gamma));

	double halfFovy = fovy * 0.5;
	double d = hypot(sz.width, sz.height);
	double sideLength = scale * d / cos(deg2Rad(halfFovy));
	double h = d / (2.0 * sin(deg2Rad(halfFovy)));
	double n = h - (d / 2.0);
	double f = h + (d / 2.0);

	Mat F = Mat(4, 4, CV_64FC1); //Allocate 4x4 transformation matrix F
	Mat Rtheta;
	Rtheta = Mat::eye(4, 4, CV_64FC1); //Allocate 4x4 rotation matrix around Z-axis by theta degrees
	Mat Rphi;
	Rphi = Mat::eye(4, 4, CV_64FC1); //Allocate 4x4 rotation matrix around X-axis by phi degrees
	Mat Rgamma;
	Rgamma = Mat::eye(4, 4, CV_64FC1); //Allocate 4x4 rotation matrix around Y-axis by gamma degrees

	Mat T;
	T = Mat::eye(4, 4, CV_64FC1); //Allocate 4x4 translation matrix along Z-axis by -h units
	Mat P;
	P = Mat::zeros(4, 4, CV_64FC1); //Allocate 4x4 projection matrix

	//Rtheta
	Rtheta.at<double>(0, 0) = Rtheta.at<double>(1, 1) = ct;
	Rtheta.at<double>(0, 1) = -st;
	Rtheta.at<double>(1, 0) = st;
	//Rphi
	Rphi.at<double>(1, 1) = Rphi.at<double>(2, 2) = cp;
	Rphi.at<double>(1, 2) = -sp;
	Rphi.at<double>(2, 1) = sp;
	//Rgamma
	Rgamma.at<double>(0, 0) = Rgamma.at<double>(2, 2) = cg;
	Rgamma.at<double>(0, 2) = -sg;
	Rgamma.at<double>(2, 0) = sg;

	//T
	T.at<double>(2, 3) = -h;
	//P
	P.at<double>(0, 0) = P.at<double>(1, 1) = 1.0 / tan(deg2Rad(halfFovy));
	P.at<double>(2, 2) = -(f + n) / (f - n);
	P.at<double>(2, 3) = -(2.0 * f * n) / (f - n);
	P.at<double>(3, 2) = -1.0;
	//Compose transformations
	F = P * T * Rphi * Rtheta * Rgamma; //Matrix-multiply to produce master matrix

	//Transform 4x4 points
	double ptsIn[4 * 3];
	double ptsOut[4 * 3];
	double halfW = sz.width / 2, halfH = sz.height / 2;

	ptsIn[0] = -halfW;
	ptsIn[1] = halfH;
	ptsIn[3] = halfW;
	ptsIn[4] = halfH;
	ptsIn[6] = halfW;
	ptsIn[7] = -halfH;
	ptsIn[9] = -halfW;
	ptsIn[10] = -halfH;
	ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0; //Set Z component to zero for all 4 components

	Mat ptsInMat(1, 4, CV_64FC3, ptsIn);
	Mat ptsOutMat(1, 4, CV_64FC3, ptsOut);

	perspectiveTransform(ptsInMat, ptsOutMat, F); //Transform points

	//Get 3x3 transform and warp image
	Point2f ptsInPt2f[4];
	Point2f ptsOutPt2f[4];

	for (int i = 0; i < 4; i++) {
		Point2f ptIn(ptsIn[i * 3 + 0], ptsIn[i * 3 + 1]);
		Point2f ptOut(ptsOut[i * 3 + 0], ptsOut[i * 3 + 1]);
		ptsInPt2f[i] = ptIn + Point2f(halfW, halfH);
		ptsOutPt2f[i] = (ptOut + Point2f(1, 1)) * (sideLength * 0.5);
	}

	M = getPerspectiveTransform(ptsInPt2f, ptsOutPt2f);

	//Load corners vector
	if (corners) {
		corners->clear();
		corners->push_back(ptsOutPt2f[0]); //Push Top Left corner
		corners->push_back(ptsOutPt2f[1]); //Push Top Right corner
		corners->push_back(ptsOutPt2f[2]); //Push Bottom Right corner
		corners->push_back(ptsOutPt2f[3]); //Push Bottom Left corner
	}
}

void warpImage(const Mat &src, double theta, double phi, double gamma,
		double scale, double fovy, Mat& dst, Mat& M, vector<Point2f> &corners) {
	double halfFovy = fovy * 0.5;
	double d = hypot(src.cols, src.rows);
	double sideLength = scale * d / cos(deg2Rad(halfFovy));

	warpMatrix(src.size(), theta, phi, gamma, scale, fovy, M, &corners); //Compute warp matrix
	//warpPerspective(src, dst, M, Size(sideLength, sideLength)); //Do actual image warp
	warpPerspective(src, dst, M, Size(sideLength, sideLength)); //Do actual image warp
	int w = src.cols, h = src.rows;
	dst = Mat(dst,
			Rect((int) ((sideLength - w) / 2), (int) ((sideLength - h) / 2), w,
					h));
}

void applyDistortion(const Mat &src, double deviation) {

}

void alterImage(const Mat &src, Mat &dest, double rot_x, double rot_y, double rot_z,
		double noise_deviation = 0, double brightness_max_deviation = 0) {
	Mat warp;
	vector<Point2f> corners;
	warpImage(src, rot_z, rot_x, rot_y, 1, 5, dest, warp, corners);

}

Mat image;
int max_rot_x, max_rot_y, max_rot_z, desired_noise_deviation, max_brightness_change;
int rot_x_steps, rot_y_steps, rot_z_steps, noise_steps, brightness_change_steps;

void on_config_change( int, void* ){


}


int main(int argc, char *argv[]) {

	if (argc < 2) {
		cout << endl << "UNKNOWN PARAMETERS. Please supply at least the file name."
				<< endl << endl;
		cout << " --file   :   filename -> process this single file." << endl;
		cout << "OR:" << endl;
		cout << " --dir    :   Process all files in specified directory."
				<< endl << endl;
		cout << "OPTIONAL:" << endl;
		cout
				<< " --output :   (optional) output directory (default -> current directory)"
				<< endl << endl;
		exit(-1);
	}

	char file[128], directory[128], output[128];

	for (int i = 1; i < argc; i+=2) {
		if(!strcoll(argv[i], "--file")){
			cout << "file: " << argv[i+1] << endl;
			strcpy(file, argv[i+1]);

		} else if(!strcoll(argv[i], "--dir")){
			cout << "dir: "<< argv[i+1] << endl;
			strcpy(directory, argv[i+1]);

		} else if(!strcoll(argv[i], "--output")){
			cout << "output: " << argv[i+1] << endl;
			strcpy(output, argv[i+1]);

		} else {
			cout << "Invalid Argument: " << argv[i];
			exit(-2);
		}
	}

	image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	image.convertTo(image, CV_32F, 1 / 255.0);



	// CONFIG PHASE

	namedWindow("Configure Sample Multiplier", WINDOW_AUTOSIZE);
	imshow("Configure Sample Multiplier", image);



	// HIER WEITERMACHEN:
	createTrackbar("Max. X-Rotation", "Configure Sample Multiplier", &max_rot_x, 180, on_config_change, NULL);
	createTrackbar("Max. Y-Rotation", "Configure Sample Multiplier", &max_rot_y, 180, on_config_change, NULL);
	createTrackbar("Max. Z-Rotation", "Configure Sample Multiplier", &max_rot_z, 180, on_config_change, NULL);
	createTrackbar("Desired Noise Deviation", "Configure Sample Multiplier", &desired_noise_deviation, 100, on_config_change, NULL);
	createTrackbar("Max Brightness Change", "Configure Sample Multiplier", &max_brightness_change, 180, on_config_change, NULL);

	createTrackbar("", "Configure Sample Multiplier", &max_rot_x, 180, on_config_change, NULL);
	createTrackbar("Max. X-Rotation", "Configure Sample Multiplier", &max_rot_x, 180, on_config_change, NULL);
	createTrackbar("Max. X-Rotation", "Configure Sample Multiplier", &max_rot_x, 180, on_config_change, NULL);
	createTrackbar("Max. X-Rotation", "Configure Sample Multiplier", &max_rot_x, 180, on_config_change, NULL);
	createTrackbar("Max. X-Rotation", "Configure Sample Multiplier", &max_rot_x, 180, on_config_change, NULL);


	Mat resultImg;

	alterImage(image, resultImg, 0, 0, 0);

	cout << "source size = " << image.cols << "|" << image.rows << endl;
	cout << "dest size = " << resultImg.cols << "|" << resultImg.rows << endl;

	namedWindow("Destination", WINDOW_AUTOSIZE);
	imshow("Destination", resultImg);

	waitKey(0);

}
