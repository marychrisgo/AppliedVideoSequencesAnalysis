/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 * 
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

// dataset:
// ./main /home/sebasmos/AVSA2020datasets/AVSA_Lab3_datasets/dataset_lab3

//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

//Header ShowManyImages
#include "ShowManyImages.hpp"

//include for blob-related functions
#include "blobs.hpp"

#include "opencv2/video/tracking.hpp"


//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

#define MIN_WIDTH 10
#define MIN_HEIGHT 10

cv::KalmanFilter KF;
Mat_<float> measurement(2,1); 
int type = 1;


std::vector<Point> measurement_vect;
std::vector<Point> state_pre_vect;
std::vector<Point> state_post_vect;

std::vector<Point> state_pre_copy;
std::vector<Point> state_post_copy;

//------------------------DRAWING------------------------------------------------------
// Store measurements, state_pre, state_post (Point type)

Mat Draw_tracking(cv::Mat result, cv::Point_<float> measurement, cv::Point_<float> state_pre, cv::Point_ <float> state_post, bool flag  ){

	Scalar red = Scalar (0, 0, 255);

	Scalar blue = Scalar (255, 0, 0);

	Scalar green = Scalar  (163, 73, 164);

	Mat final_img;

	int radius= 5;

	result.copyTo(final_img);

	// Store results in vectors for plotting purposes.
	measurement_vect.push_back(measurement);
	state_pre_vect.push_back(state_pre);
	state_post_vect.push_back(state_post);

	cout << "kalman prediction (state_pre) aqui: " << state_pre.x << " " << state_pre.y << endl;
	cout << "kalman corrected state (state_post): " << state_post.x << " " << state_post.y << endl;
			
	if (flag!=true | state_pre_vect.size() !=0 | state_post_vect.size() !=0 ){
		
	
		for (size_t i = 0; i < state_pre_vect.size () - 1; i++) {

		circle (final_img, state_pre_vect[i], 5, blue, 2);
		line (final_img, state_pre_vect[i], state_pre_vect[i+1], blue, 1);
		putText(final_img, "state_pre_vect: ", 
									cv::Point(10, 15),
									FONT_HERSHEY_SIMPLEX, 
									0.5,
									blue);
		state_pre_copy = state_pre_vect;
		}
		for (size_t i = 0; i < state_post_vect.size () - 1; i++) {

		circle (final_img, state_post_vect[i], 5, green, 2);
		line (final_img, state_post_vect[i], state_pre_vect[i+1], green, 1);
		putText(final_img, "state_post_vect: ", 
									cv::Point(10, 45),
									FONT_HERSHEY_SIMPLEX, 
									0.5,
									green);

		state_post_copy = state_post_vect;
		}
	
	//-------------------plot measurement for comparaison------------

	
	for (Point point : measurement_vect) {

		line (final_img, Point (point.x - radius / 2, point.y), Point (point.x + radius / 2, point.y), red, 1);

		line (final_img, Point (point.x, point.y - radius / 2), Point (point.x, point.y + radius / 2), red, 1);

	}

	for (size_t i = 0; i < measurement_vect.size () - 1; i++) {

		//cout << measurement_vect[i]<< endl;
		//circle (final_img, measurement_vect[i], 5, red, 2);
		line (final_img, measurement_vect[i], measurement_vect[i+1], red, 1);
		putText(final_img, "measurement_vect: ", 
									cv::Point(10, 70),
									FONT_HERSHEY_SIMPLEX, 
									0.5,
									red);
	}

	}
	else{
		if (state_post_copy.size()!=0 | state_pre_vect.size() !=0 | state_post_vect.size() !=0 ){
			for (size_t i = 0; i < state_pre_copy.size () - 1; i++) {

				circle (final_img, state_pre_copy[i], 5, blue, 2);
				line (final_img, state_pre_copy[i], state_pre_vect[i+1], blue, 1);
				putText(final_img, "state_pre_vect: ", 
											cv::Point(10, 15),
											FONT_HERSHEY_SIMPLEX, 
											0.5,
											blue);
				}
				for (size_t i = 0; i < state_post_copy.size () - 1; i++) {

				circle (final_img, state_post_copy[i], 5, green, 2);
				line (final_img, state_post_vect[i], state_pre_vect[i+1], green, 1);
				putText(final_img, "state_post_vect: ", 
											cv::Point(10, 45),
											FONT_HERSHEY_SIMPLEX, 
											0.5,
											green);
				}
				
			
		}
	}		
	return final_img;

}

	
void Kalman_initialization (int type, float x, float y)
{
	if (type == 0) { // velocity
		KF.init(4, 2, 0);
		// x,y - biggest blob center
		measurement = Mat_<float>::zeros(2,1);
		/*
		measurement.at<float>(0, 0) = x;
		measurement.at<float>(0, 0) = y;
		*/
		/*
		// TO CHECK
		KF.statePre.setTo(0);
		KF.statePre.at<float>(0, 0) = x;
		KF.statePre.at<float>(1, 0) = y;
		// TO CHECK
		KF.statePost.setTo(0);
		KF.statePost.at<float>(0, 0) = x;
		KF.statePost.at<float>(1, 0) = y; 

		*/
		//----------------------------------------------------------------------------------------------------------
		// INITIALIZE KF MATRIXES
		//----------------------------------------------------------------------------------------------------------
 		// A MATRIX - TRANSITION MATRIX
		KF.transitionMatrix = (Mat_<float>(4,4) <<  1, 1, 0, 0, 
													0, 1, 0, 0,
													0, 0, 1, 1, 
													0, 0, 0, 1); 
		// Q MATRIX - COVARIANCE MATRIX - ERROR COVARIANCE
		KF.processNoiseCov = (Mat_<float>(4,4) <<
			25, 0, 0, 0,
			0, 10, 0, 0,
			0, 0, 20, 0,
			0, 0, 0, 10); 
		// H MATRIX - MEASUREMENT MATRIX
		KF.measurementMatrix = (Mat_<float>(2,4) << 1, 0, 0, 0, 
													0, 0, 1, 0); 
		// R MATRIX - MEASUREMENT NOISE COVARIANCE MATRIX OR COVARIANCE OF OBSERVATION ERROR
		KF.measurementNoiseCov = (Mat_<float>(2,2) <<
												2, 0,
												0, 2);
												//25, 0,
												//0, 25); 
		// PROCESS NOISE COVARIANCE P 
		setIdentity(KF.errorCovPost, Scalar::all(1000000)); // P : posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
		//KF.errorCovPost = (Mat_<float>(4,4) <<
		//										1000000,0,0,0,
		//										0,1000000,0,0,
		//										0,0,1000000,0,
		//										0,0,0,1000000
		//										);// P Covariance 
	}

	else if (type == 1){//acceleration
		KF.init (6,2,0);
		// x,y - biggest blob center
		measurement = Mat_<float>::zeros(2,1); // Position x,y
		measurement.at<float>(0, 0) = x;
		measurement.at<float>(0, 0) = y;
		// TO CHECK
		KF.statePre.setTo(0);
		KF.statePre.at<float>(0, 0) = x;
		KF.statePre.at<float>(1, 0) = y;
		// TO CHECK
		KF.statePost.setTo(0);
		KF.statePost.at<float>(0, 0) = x;
		KF.statePost.at<float>(1, 0) = y;
		// A MATRIX - TRANSITION MATRIX
		KF.transitionMatrix = (Mat_<float>(6,6) <<
			1, 0, 1, 0, 0.5, 0,
			0, 1, 0, 1, 0, 0.5,
			0, 0, 1, 0, 1, 0,
			0, 0, 0, 1, 0, 1,
			0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 1); 
		// Q MATRIX - COVARIANCE MATRIX
		KF.processNoiseCov = (Mat_<float>(6,6) <<
				25, 0, 0, 0, 0 , 0,
				0, 25, 0, 0, 0, 0,
				0, 0, 10, 0, 0, 0,
				0, 0, 0, 10, 0, 0,
				0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 1); // Q matrix, noise covariance matrix
		KF.measurementMatrix = (Mat_<float>(2,6) <<
			1, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0); // H matrix, measurement matrix
		KF.measurementNoiseCov = (Mat_<float>(2,2) <<
			0, 0,
			0, 0); // R matrix, measurement noise covariance matrix
		// Prior Covariance P -  gets close to zero if we trust prediction
		KF.errorCovPost = (Mat_<float>(6,6) <<
										0.0001,0,0,0,0,0,
												0,0.0001,0,0,0,0,
												0,0,0.0001,0,0,0,
												0,0,0,0.0001,0,0,
												0,0,0,0,0.0001,0,
												0,0,0,0,0,0.0001
												// P Covariance 
												);//
												
		cout <<"SDFSDFSDSDFSDFSFDSFDFSDFSDFDSFDS" <<endl;
		
	}
    //setIdentity(KF.errorCovPost, Scalar::all(1)); // P : posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)

}
	
Point kalmanPredict() 
{
    Mat prediction = KF.predict();
    Point predictPt(prediction.at<float>(0),prediction.at<float>(2)); // TO CHECK: indexes for vx,vy, and x,y
    return predictPt;
}

Point kalmanCorrect(float x, float y)
{
    measurement(0) = x;
    measurement(1) = y;
    Mat estimated = KF.correct(measurement);
    Point statePt(estimated.at<float>(0),estimated.at<float>(2));
    return statePt;
}


//main function
int main(int argc, char ** argv) 
{

	std::cout << "dataset path " << argv[1] << "\n";

	//Point meas;
	Mat frame; // current Frame
	Mat fgmask; // foreground mask
	Mat fgmask_filtered;
	Point2f meas;
	
	int radius= 20;
	Mat tracking_Result; // to add lines and circles
	std::vector<cvBlob> bloblist; // list for blobs
	std::vector<cvBlob> bloblistFiltered; // list for blobs

	double t, acum_t; //variables for execution time
	int t_freq = getTickFrequency();
	int index;

		string dataset_path = argv[1]; //"/home/sebasmos/AVSA2020datasets/AVSA_Lab3_datasets/dataset_lab3"; //SET THIS DIRECTORY according to your download
		string dataset_cat[1] = {"dataset_lab3"};
		string baseline_seq[1] = {"lab3.3/abandonedBox_600_1000_clip.mp4"};
		string image_path = "";

		string project_root_path = "/home/sebasmos/AVSA2020results"; //SET THIS DIRECTORY according to your project
		string project_name = "kalman"; //SET THIS DIRECTORY according to your project
		string results_path = project_root_path+"/"+project_name+"/results";

		int NumCat = sizeof(dataset_cat)/sizeof(dataset_cat[0]); //number of categories (have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

		//Loop for all categories
		for (int c=0; c<NumCat; c++ )
		{
			// create directory to store results for category
			string makedir_cmd = "mkdir "+results_path + "/" + dataset_cat[c];
			system(makedir_cmd.c_str());

			int NumSeq = sizeof(baseline_seq)/sizeof(baseline_seq[0]);  //number of sequences per category ((have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

			//Loop for all sequence of each category
			for (int s=0; s<NumSeq; s++ )
			{
			
			VideoCapture cap;//reader to grab videoframes

			//Compose full path of images
			string inputvideo = dataset_path + "/" + dataset_cat[c] + "/" + baseline_seq[s] + image_path;
			cout << "Accessing sequence at " << inputvideo << endl;

			//open the video file to check if it exists
			cap.open(inputvideo);
			if (!cap.isOpened()) {
				cout << "Could not open video file " << inputvideo << endl;
			return -1;
			}

			// create directory to store results for sequence
			string makedir_cmd = "mkdir "+results_path + "/" + dataset_cat[c] + "/" + baseline_seq[s];
			system(makedir_cmd.c_str());

			//MOG2 approach

            Ptr<BackgroundSubtractorMOG2>  pMOG2 = cv::createBackgroundSubtractorMOG2();
			
			// Set threshold
			pMOG2->setVarThreshold(16);
			pMOG2->setHistory(50);

			//main loop
			Mat img; // current Frame

			int it=1;
			acum_t=0;

			//----------------------------------------------------------------------------------------------------------
			// KALMAN INIT
			//----------------------------------------------------------------------------------------------------------
			
			Point state_post, state_pre;
			Kalman_initialization(0,0,0);

			//----------------------------------------------------------------------------------------------------------
			// END KALMAN INIT
			//----------------------------------------------------------------------------------------------------------
			
			
			for (;;) {
				//get frame
				cap >> img;

				//check if we achieved the end of the file (e.g. img.data is empty)
				if (!img.data)
					break;

				//Time measurement
				t = (double)getTickCount();

				//apply algs: Get current img & store in frame
				img.copyTo(frame);				
				
				// Compute fgmask
				double learningrate=0.01;
				pMOG2->apply(frame, fgmask, learningrate); 
				//----------------------------------------------------------------------------------------------------------
				// APPLY FILTER
				//----------------------------------------------------------------------------------------------------------
				//
				Mat kernel = getStructuringElement(MORPH_RECT, Size(10,10)); // https://docs.opencv.org/3.4/df/d5e/samples_2cpp_2tutorial_code_2ImgProc_2Morphology_1_8cpp-example.html#a16 
				
                morphologyEx(fgmask, fgmask_filtered, MORPH_OPEN , kernel);
				
				//----------------------------------------------------------------------------------------------------------
				// END OF FILTER
				//----------------------------------------------------------------------------------------------------------

				int connectivity = 8; // 4 or 8

				// Extract the blobs in fgmask

				extractBlobs(fgmask_filtered, bloblist, connectivity);	
				
				removeSmallBlobs(bloblist, bloblistFiltered, MIN_WIDTH, MIN_HEIGHT);	
				//----------------------------------------------------------------------------------------------------------
				//-----------------------DETECT THE CENTER OF THE BIGGEST BLOB----------------------------------------------
				//----------------------------------------------------------------------------------------------------------
				int val = biggest_blob(bloblistFiltered);				
				//------------------------END OF BIGGEST BLOB DETECTION-----------------------------------------------------

				frame.copyTo(tracking_Result);

				int ii = 0;
				bool flag = false;
				int x;
				int y;			


				if (bloblistFiltered.empty()){
					//meas = 0;
					ii = ii+1;
					cout << "No blobs detected: "<< bloblistFiltered.size() << endl;
					if (ii<10){
						state_pre = kalmanPredict();	
						flag = false;					
					}
					state_pre = kalmanPredict();
					state_pre.x = x;
					state_pre.y = y;	
					flag = true;	
				}else{
					meas = Point2f(bloblistFiltered[val].x + bloblistFiltered[val].w/2,  bloblistFiltered[val].y + bloblistFiltered[val].h/2); 
				
					x = bloblistFiltered[val].x + bloblistFiltered[val].w/2;
					y = bloblistFiltered[val].y + bloblistFiltered[val].h/2;
					// PREDICTION
					state_pre = kalmanPredict();
					//cout << "kalman prediction (state_pre): " << state_pre.x << " " << state_pre.y << endl;
					measurement(0) = x;
					measurement(1) = y;

					// CORRECTION
					state_post = kalmanCorrect(x, y);					
				}			
									
					// if flag is true, means that no object was detected for at least 10 frames, so we are sure about the measurement, therefore make prediction = measurement
				if (flag==true){
					state_pre.x = x;
					state_pre.y = y;				
				}
				//Time measurement
				t = (double)getTickCount() - t;
        		        //if (_CONSOLE_DEBUG) cout << "proc. time = " << 1000*t/t_freq << " milliseconds."<< endl;
				acum_t=+t;

				//SHOW RESULTS
				//get the frame number and write it on the current frame

				string title= project_name + " | Frame - FgM - Tracking | Blobs - Classes - Stat Classes | BlobsFil - ClassesFil - Stat ClassesFil | ("+dataset_cat[c] + "/" + baseline_seq[s] + ")";

				//ShowManyImages(title, 6, frame, fgmask, result,
				//		paintBlobImage(frame,bloblistFiltered, true), paintBlobImage(frame,bloblistFiltered, true), paintBlobImage(frame,sbloblistFiltered, true));
				ShowManyImages(title, 6, frame, fgmask, Draw_tracking(tracking_Result,  meas,state_pre, state_post, flag),
						paintBlobImage(frame,bloblistFiltered, true), paintBlobImage(frame,bloblistFiltered, true), paintBlobImage(frame,bloblistFiltered, true));
				
				imshow("result: ",Draw_tracking(tracking_Result,  meas,state_pre,state_post, flag));
				//exit if ESC key is pressed
				if(waitKey(20) == 27) break;
				//---------------------------------------------------------
				char key = waitKey(5); // waits to display frame
				if (key == 'q')
					break;
				//---------------------------------------------------------
				it++;
			} //main loop

	cout << it-1 << "frames processed in " << 1000*acum_t/t_freq << " milliseconds."<< endl;


	//release all resources

	cap.release();
	destroyAllWindows();
	waitKey(0); // (should stop till any key is pressed .. doesn't!!!!!)
}
}
return 0;
}


/**
 * REFERENCES
 * https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-using-kalman-filter/ 
 * https://stackoverflow.com/questions/18403918/opencv-kalman-filter-prediction-without-new-observtion 
 * https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc 
 * 
 */
