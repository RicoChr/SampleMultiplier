#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

static double rad2Deg(double rad) {
	return rad * (180 / M_PI);
} //Convert radians to degrees

static double deg2Rad(double deg) {
	return deg * (M_PI / 180);
} //Convert degrees to radians


void printType(Mat &mat) {
         if(mat.depth() == CV_8U)  printf("unsigned char(%d)", mat.channels());
    else if(mat.depth() == CV_8S)  printf("signed char(%d)", mat.channels());
    else if(mat.depth() == CV_16U) printf("unsigned short(%d)", mat.channels());
    else if(mat.depth() == CV_16S) printf("signed short(%d)", mat.channels());
    else if(mat.depth() == CV_32S) printf("signed int(%d)", mat.channels());
    else if(mat.depth() == CV_32F) printf("float(%d)", mat.channels());
    else if(mat.depth() == CV_64F) printf("double(%d)", mat.channels());
    else                           printf("unknown(%d)", mat.channels());
}


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
	dst = Mat(dst, Rect((int) ((sideLength - w) / 2), (int) ((sideLength - h) / 2), w, h));
}

void applyNoise(Mat &image, double deviation) {
	cv::Mat noise(image.size(), image.type());
	double m = (0);
	double sigma = (deviation / 255.0);
	cv::randn(noise, m, sigma);
	image += noise;
}

void applyBrightnessChange(Mat &image, float brightness_change){
	image += brightness_change;
}

void scaleImage(const Mat &src, Mat &dest, double scale){
	Mat warp;
	vector<Point2f> corners;
	warpImage(src, 0, 0, 0, scale, 5, dest, warp, corners);
}

void alterImage(const Mat &src, Mat &dest, double rot_x, double rot_y, double rot_z,
		double noise_deviation = 0, double brightness_max_deviation = 0) {
	Mat warp;
	vector<Point2f> corners;
	warpImage(src, rot_z, rot_x, rot_y, 1, 5, dest, warp, corners);
	applyNoise(dest, noise_deviation);
	applyBrightnessChange(dest, brightness_max_deviation);

}


Mat suppliedImage;
vector<Mat> suppliedImages;
int padding;
int max_rot_x, max_rot_y, max_rot_z, desired_noise_deviation, max_brightness_change;
int rot_x_steps, rot_y_steps, rot_z_steps, noise_steps, brightness_change_steps, unusedVal;
//int display_scale;

const char * CONFIG_WINDOW_TITLE = "Configure Sample Multiplier";
const char * NEGATIVE_EXAMPLE_WINDOW_TITLE = "Negative Value Example";

bool hasFile = false, hasDir = false, hasOutput = false;

void on_config_change( int, void* ){

	int samples_to_be_generated = (
				rot_x_steps
				+ (rot_x_steps*rot_y_steps)
				+ ((rot_x_steps + rot_x_steps*rot_y_steps)*rot_z_steps)
				+ ((rot_x_steps + (rot_x_steps*rot_y_steps) + ((rot_x_steps + rot_x_steps*rot_y_steps)*rot_z_steps))*noise_steps)
				+ ((rot_x_steps	+ (rot_x_steps*rot_y_steps)	+ ((rot_x_steps + rot_x_steps*rot_y_steps)*rot_z_steps)	+ ((rot_x_steps + (rot_x_steps*rot_y_steps) + ((rot_x_steps + rot_x_steps*rot_y_steps)*rot_z_steps))*noise_steps))*brightness_change_steps)
				) * suppliedImages.size()		;

	cout << endl << "GENERATING " << samples_to_be_generated << " ALTERED IMAGES" << endl << endl;
	Mat displayImagePos, displayImageNeg;
	Mat m;
	//displayImage = demoImage;
	//if(display_scale == 0) display_scale = 1;
	//scaleImage(demoImage, displayImage, ((double)display_scale));

	if(hasDir)suppliedImage = suppliedImages.at(0).clone();


	if(padding > (suppliedImage.cols/2 - 1) || (padding > (suppliedImage.rows/2 - 1))){
		padding = std::min(suppliedImage.cols/2 - 1, suppliedImage.rows/2 - 1);
	}

	alterImage(suppliedImage, displayImagePos, max_rot_x, max_rot_y, max_rot_z, desired_noise_deviation, max_brightness_change/255.0);
	cvtColor(displayImagePos, displayImagePos, CV_GRAY2BGR);
	rectangle(displayImagePos, Point(padding, padding), Point(displayImagePos.cols-1-padding, displayImagePos.rows-1-padding), Scalar(0,255,0));
	imshow(CONFIG_WINDOW_TITLE, displayImagePos);
	alterImage(suppliedImage, displayImageNeg, -max_rot_x, -max_rot_y, -max_rot_z, 0, -max_brightness_change/255.0);
	imshow(NEGATIVE_EXAMPLE_WINDOW_TITLE, displayImageNeg);
}




void generateAlteredImages(Mat image, vector<Mat> &generatedImages){
	cv::Mat sizeMat(Size(image.cols, image.rows), image.type());

		for(int i = 1; i <= rot_x_steps; i++){
			Mat newImage;
			alterImage(image, newImage, (-max_rot_x/2.0) + (i*max_rot_x/rot_x_steps), 0, 0, 0, 0);
			generatedImages.push_back(newImage);
		}

		int sizebefore = generatedImages.size();
		for(int i = 1; i <= rot_y_steps; i++){
			for(int j = 0; j < sizebefore; j++){
				Mat newImage;
				alterImage(generatedImages.at(j), newImage, 0, (-max_rot_y/2.0) + (i*max_rot_y/rot_y_steps), 0, 0, 0);
				generatedImages.push_back(newImage);
			}
		}

		sizebefore = generatedImages.size();
		for(int i = 1; i <= rot_z_steps; i++){
			for(int j = 0; j < sizebefore; j++){
				Mat newImage;
				alterImage(generatedImages.at(j), newImage, 0, 0, (-max_rot_z/2.0) + (i*max_rot_z/rot_z_steps), 0, 0);
				generatedImages.push_back(newImage);
			}
		}

		sizebefore = generatedImages.size();
		for(int i = 1; i <= noise_steps; i++){
			for(int j = 0; j < sizebefore; j++){
				Mat newImage;
				alterImage(generatedImages.at(j), newImage, 0, 0, 0, i*desired_noise_deviation/noise_steps, 0);
				generatedImages.push_back(newImage);
			}
		}

		sizebefore = generatedImages.size();
		for(int i = 1; i <= brightness_change_steps; i++){
			for(int j = 0; j < sizebefore; j++){
				Mat newImage;
				alterImage(generatedImages.at(j), newImage, 0, 0, 0, 0, i*max_brightness_change/brightness_change_steps/255.0);
				generatedImages.push_back(newImage);
			}
		}
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
				<< " --output :   (optional) output directory (default -> current directory)" << endl
				<< " --show :   (optional) display output before saving."
				<< endl << endl;
		exit(-1);
	}

	char file[128], directory[128], output[128];


	uint8_t show_flag = 0;

	for (int i = 1; i < argc; i+=2) {
		if(!strcoll(argv[i], "--file")){
			cout << "file: " << argv[i+1] << endl;
			strcpy(file, argv[i+1]);
			hasFile = true;

		} else if(!strcoll(argv[i], "--dir")){
			cout << "dir: "<< argv[i+1] << endl;
			strcpy(directory, argv[i+1]);
			hasDir = true;

		} else if(!strcoll(argv[i], "--output")){
			cout << "output: " << argv[i+1] << endl;
			strcpy(output, argv[i+1]);
			hasOutput = true;

		} else if(!strcoll(argv[i], "--show")){
			cout << "output: " << argv[i+1] << endl;
			show_flag = 1;
		} else {
			cout << "Invalid Argument: " << argv[i];
			exit(-2);
		}
	}

	if(hasFile && hasDir){
		cout << "ERROR: Please supply either the filename or the directory. Not both." << endl;
		exit(-3);
	}


	if(hasDir){
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir (directory)) != NULL) {
		  /* print all the files and directories within directory */
		  while ((ent = readdir (dir)) != NULL) {
			printf ("%s\n", ent->d_name);
			if(strstr(ent->d_name, ".png") != NULL || strstr(ent->d_name, ".jpg") != NULL || strstr(ent->d_name, ".jpeg") != NULL){
				Mat loadedImage = imread((string(directory) + string(ent->d_name)).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
				loadedImage.convertTo(loadedImage, CV_32F, 1 / 255.0);
				suppliedImages.push_back(loadedImage);
			}
		  }
		  closedir (dir);
		} else {
		  /* could not open directory */
		  perror ("");
		  return EXIT_FAILURE;
		}
		cout << "Read " << suppliedImages.size() << " images." << endl;
	} else {
		suppliedImage = imread(file, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
		suppliedImage.convertTo(suppliedImage, CV_32F, 1 / 255.0);
	}



	// CONFIG PHASE

	namedWindow(CONFIG_WINDOW_TITLE, 0);
	namedWindow(NEGATIVE_EXAMPLE_WINDOW_TITLE, 0);

	if(hasFile){
		imshow(CONFIG_WINDOW_TITLE, suppliedImage);
		imshow(NEGATIVE_EXAMPLE_WINDOW_TITLE, suppliedImage);
	} else {
		imshow(CONFIG_WINDOW_TITLE, suppliedImages.at(0));
		imshow(NEGATIVE_EXAMPLE_WINDOW_TITLE, suppliedImages.at(0));
	}

	createTrackbar("Padding", CONFIG_WINDOW_TITLE, &padding, 512, on_config_change, NULL);
	createTrackbar("Max. X-Rotation", CONFIG_WINDOW_TITLE, &max_rot_x, 180, on_config_change, NULL);
	createTrackbar("Max. Y-Rotation", CONFIG_WINDOW_TITLE, &max_rot_y, 180, on_config_change, NULL);
	createTrackbar("Max. Z-Rotation", CONFIG_WINDOW_TITLE, &max_rot_z, 180, on_config_change, NULL);
	createTrackbar("Desired Noise Deviation", CONFIG_WINDOW_TITLE, &desired_noise_deviation, 255, on_config_change, NULL);
	createTrackbar("Max Brightness Change", CONFIG_WINDOW_TITLE, &max_brightness_change, 255, on_config_change, NULL);

	createTrackbar("X-Rotation Alterations", CONFIG_WINDOW_TITLE, &rot_x_steps, 30, on_config_change, NULL);
	createTrackbar("Y-Rotation Alterations", CONFIG_WINDOW_TITLE, &rot_y_steps, 30, on_config_change, NULL);
	createTrackbar("Z-Rotation Alterations", CONFIG_WINDOW_TITLE, &rot_z_steps, 30, on_config_change, NULL);
	createTrackbar("Noise Alterations", CONFIG_WINDOW_TITLE, &noise_steps, 30, on_config_change, NULL);
	createTrackbar("Brightness Alterations", CONFIG_WINDOW_TITLE, &brightness_change_steps, 30, on_config_change, NULL);
	createTrackbar("unused", NEGATIVE_EXAMPLE_WINDOW_TITLE, &unusedVal, 1, NULL, NULL);

	int run_flag = 0;
	createTrackbar("run", CONFIG_WINDOW_TITLE, &run_flag, 1, NULL, NULL);

	while(run_flag == 0) waitKey(20);

	destroyAllWindows();




	// GENERATION PHASE

	vector<Mat> all_generated_samples;
	if(hasFile){
		generateAlteredImages(suppliedImage, all_generated_samples);
	} else {
		cout << "Generating...       ";
		int count = suppliedImages.size();
		int i = 1;

		for(vector<Mat>::iterator it = suppliedImages.begin(); it != suppliedImages.end(); it++){

			int percentage = 100 * i / count;
			printf("\033[4D");
			printf("%3.0d\%", percentage);
			fflush(stdout);

			vector<Mat> generated_samples;
			generateAlteredImages(*it, generated_samples);
			all_generated_samples.insert(all_generated_samples.end(), generated_samples.begin(), generated_samples.end());
			i++;
		}
	}
	printf("\n");

	for(uint32_t i = 0; i < all_generated_samples.size(); i++){
		Mat converted;
		converted = all_generated_samples.at(i);
		converted *= 255.0;
		all_generated_samples.at(i).convertTo(converted, CV_16UC3, 255.0);
		if(show_flag == 1){
			imshow("Saving this image.", converted);
			waitKey(0);
		}

		mkdir(output, 0777);

		converted = converted(Rect(padding, padding, converted.cols-(padding*2), converted.rows-(padding*2)));

		if(hasOutput) 	imwrite("./" + string(output) + "/sample_" + std::to_string(i) + ".png", converted);
		else			imwrite("./output/sample_" + std::to_string(i) + ".png", converted);
	}



}
