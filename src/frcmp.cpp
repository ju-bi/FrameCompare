#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur

#include <iostream>
#include <string>

using namespace cv;



/*
 * Usage:
 *
 *   frcmp <path_to_reference_frame> <path_to_frame_to_compare> [--quiet <mse_threshold> <ssi_threshold> [<path_to save_diff_frame>]]
 * 
 * Exit code (quiet mode only):
 * 
 *   2 if error in command-lines arguments (print usage)
 *   1 if frames are not similar, according to specified thresholds (quiet mode only)
 *   0 otherwise
 *
 * References (MSE and SSI using OpenCV):
 * http://stackoverflow.com/questions/4196453/simple-and-fast-method-to-compare-images-for-similarity
 * http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
 * http://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html#videoinputpsnrmssim
 */



 // Mean Squared Error
double computeMSE(const Mat& frame0, const Mat& frame1)
{
	double errorL2 = norm(frame0, frame1, CV_L2);			// Calculate the L2 relative error between images
	return errorL2 / (double)(frame0.rows * frame0.cols);	// Convert to a reasonable scale, since L2 error is summed across all pixels of the image
}



// Structural Similarity Index
Scalar computeSSI(const Mat& frame0, const Mat& frame1)
{
	// default settings
	const double C0 = 6.5025, C1 = 58.5225;

	// frame data conversion (cannot calculate on one byte large values)
	Mat f0, f1;
	frame0.convertTo(f0, CV_32F);
	frame1.convertTo(f1, CV_32F);
	Mat f0_2 = f0.mul(f0);		// f0^2
	Mat f1_2 = f1.mul(f1);		// f1^2
	Mat f0_f1 = f0.mul(f1);		// f1*f2

	// preliminary computing
	Mat mu0, mu1;
	GaussianBlur(f0, mu0, Size(11, 11), 1.5);
	GaussianBlur(f1, mu1, Size(11, 11), 1.5);
	Mat mu0_1 = mu0.mul(mu0);
	Mat mu0_0 = mu1.mul(mu1);
	Mat mu0_mu1 = mu0.mul(mu1);

	Mat sigma0_1, sigma1_1, sigma01;
	GaussianBlur(f0_2, sigma0_1, Size(11, 11), 1.5);
	sigma0_1 -= mu0_1;
	GaussianBlur(f1_2, sigma1_1, Size(11, 11), 1.5);
	sigma1_1 -= mu0_0;
	GaussianBlur(f0_f1, sigma01, Size(11, 11), 1.5);
	sigma01 -= mu0_mu1;

	// main computing
	Mat t0, t1, t2;
	t0 = 2 * mu0_mu1 + C0;
	t1 = 2 * sigma01 + C1;
	t2 = t0.mul(t1);			// t2 = ((2*mu0_mu1 + C0).*(2*sigma01 + C1))

	t0 = mu0_1 + mu0_0 + C0;
	t1 = sigma0_1 + sigma1_1 + C1;
	t0 = t0.mul(t1);			// t1 =((mu0_1 + mu0_0 + C0).*(sigma0_1 + sigma1_1 + C1))

	Mat ssim_map;
	divide(t2, t0, ssim_map);	// ssim_map =  t2/t0;
	return mean(ssim_map);		// mssim = average of ssim map
}




// Diff
void computeDiffFrame(const Mat& frame0, const Mat& frame1, Mat& diff)
{
	// get difference in frame1 from frame0
	subtract(frame1, frame0, diff);

	// turn all different pixels to red
	CV_Assert( diff.channels() == 3); // assumes that we are using BGR images (due to IMREAD_COLOR flag)
	CV_Assert(diff.depth() == CV_8U);
	Mat ref;
	cvtColor(frame0, ref, CV_BGR2GRAY);
	cvtColor(ref, ref, CV_GRAY2BGR);
	Mat_<Vec3b> _diff = diff;
	Mat_<Vec3b> _ref = ref;
	for (int i = 0; i < diff.rows; ++i)
	{
		for (int j = 0; j < diff.cols; ++j)
		{
			if (_diff(i, j)[0] + _diff(i, j)[1] + _diff(i, j)[2] > 16) // less strict for display than 0
			{
				_diff(i, j)[0] = _ref(i, j)[0] / 4;
				_diff(i, j)[1] = _ref(i, j)[0] / 4;
				_diff(i, j)[2] = 127 + _ref(i, j)[0] / 2;
			}
			else
			{
				_diff(i, j)[0] = _ref(i, j)[0] / 2;
				_diff(i, j)[1] = _ref(i, j)[1] / 2;
				_diff(i, j)[2] = _ref(i, j)[2] / 2;
			}
		}
	}
	diff = _diff;
}




// Load 
void loadFrames(const std::string& refFramePath, const std::string& cmpFramePath, Mat& refFrame, Mat& cmpFrame, Mat& diffFrame)
{
	std::string loadErrorMessage = "";

	// load frames
	refFrame = imread(refFramePath.c_str(), IMREAD_COLOR);
	if (refFrame.empty())
		loadErrorMessage += "Failed to load reference frame '" + refFramePath + "'\n";
	
	cmpFrame = imread(cmpFramePath.c_str(), IMREAD_COLOR);
	if (cmpFrame.empty())
		loadErrorMessage += "Failed to load frame to compare '" + cmpFramePath + "'\n";
	
	// check that frames size are equal
	if (loadErrorMessage.empty() && !(refFrame.rows > 0 && refFrame.rows == cmpFrame.rows && refFrame.cols > 0 && refFrame.cols == cmpFrame.cols))
		loadErrorMessage = "Can not compare frames of different sizes [" + std::to_string(refFrame.cols) + "x" + std::to_string(refFrame.rows) 
														+ " ] <> [" + std::to_string(cmpFrame.cols) + "x" + std::to_string(cmpFrame.rows) + "]\n";

	if (!loadErrorMessage.empty())
	{
		std::cout << loadErrorMessage << std::endl;
		exit(1);
	}
}




// Usage
void printUsageAndExit()
{
	std::string usage;
	usage += "Wrong arguments:\n";
	usage += "\n";
	usage += "  frcmp <path_to_reference_frame> <path_to_frame_to_compare> [--quiet <mse_threshold> <ssi_threshold> [<path_to save_diff_frame>]]\n";
	std::cout << usage << std::endl;
	exit(2);
}




// Main
int main(int argc, char** argv)
{
	// Check arguments
	bool quietMode = (argc>3);
	if (! (argc==3 || (argc>5 && std::strcmp(argv[3],"--quiet")==0)))
	{
		printUsageAndExit();
	}
	std::string refFramePath = argv[1];
	std::string cmpFramePath = argv[2];
	std::string diffFramePath = "";
	if (argc > 6)
	{
		diffFramePath = argv[6];
	}

	// Load frames
	Mat refFrame, cmpFrame, diffFrame;
	loadFrames(refFramePath, cmpFramePath, refFrame, cmpFrame, diffFrame);

	// Compute similarity scores ([0..1] range with score 1 corresponding to perfect similarity)
	double mse_score = 1.0 - computeMSE(refFrame, cmpFrame);
	std::cout << "Similarity based on Mean Squared Error = " << mse_score << std::endl;
	Scalar ssi_3 = computeSSI(refFrame, cmpFrame);
	double ssi_score = 0.5 * (1.0 + std::min(ssi_3[0], std::min(ssi_3[1], ssi_3[2])));
	std::cout << "Similarity based on Structural Similarity Index (RGB min value)= [ " << ssi_score << " ]" << std::endl;

	// Compute diff frame, and display it if needed
	computeDiffFrame(refFrame, cmpFrame, diffFrame);
	if (!quietMode)
	{
		const char* windows[3] = { "Reference Frame", "Frame To Compare", "Diff Frame" };
		const Mat* frames[3] = { &refFrame, &cmpFrame, &diffFrame};
		for (int i = 0; i < 3; ++i)
		{
			namedWindow(windows[i], WINDOW_AUTOSIZE);	// create window
			imshow(windows[i], *frames[i]);				// show frame in window
		}
		waitKey(0); // Wait for a keystroke in the window
	}

	// Exit code
	int exitCode = 0;
	if (quietMode)
	{
		double mse_treshold = atof(argv[4]);
		double ssi_treshold = atof(argv[5]);
		if(mse_score<mse_treshold || ssi_score<ssi_treshold)
		{
			exitCode = 1;
			std::cout << "Compared frame is too different from reference frame !!" << std::endl;
			if (!diffFramePath.empty() && !imwrite(diffFramePath.c_str(), diffFrame))
			{
				std::cout << "FAILED TO WRITE DIFF FRAME TO '" + diffFramePath + "'" << std::endl;
			}
		}
	}
	return exitCode;
}
