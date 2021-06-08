//includes
#include <stdio.h> 																					//Standard I/O library
#include <numeric>																					//For std::accumulate function
#include <string> 																					//For std::to_string function
#include <opencv2/opencv.hpp>																		//opencv libraries
#include "utils.hpp" 		
#include "ShowManyImages.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/highgui.hpp>
#include "ColorTracker.hpp"																			//Header file for ColorTracker

//namespaces
using namespace cv;
using namespace std;
using namespace tracking;

//main function
int main(int argc, char ** argv)
{
	//Path for dataset and result folder
	std::string dataset_path = "/home/sebasmos/Documentos/AVSA - applied video sequences Analysis/LAB4/code/code/datasets";		//dataset location.
	std::string output_path = "/home/sebasmos/Documentos/AVSA - applied video sequences Analysis/MaryChrisGo_SebastianCajasOrdonez_Lab4/4.1 Color-based/outputs";							//location to save output videos

	// Dataset paths
	std::string sequences[] = {"car1"};											//test data

	std::string image_path = "%08d.jpg"; 															//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 												//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);											//number of sequences

	/***Loop for all sequence of each category***/
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;																					//current Frame
		int frame_idx=0;																			//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;												//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;																//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path;		 	//path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);																// reader to grab frames from videofile

		//check if video exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); 					//error if not possible to read videofile

		// Define the codec and create VideoWriter object
		//The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));	//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",						//xvid compression (cannot be changed in OpenCV)
							VideoWriter::fourcc('X','V','I','D'),10, frame_size);

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;	//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); 									 	//read groundtruth bounding boxes


		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;
		bool frame1 = true;
		
		// parameters
		int feat =6;
		int stride =9;																				
		int bins = 10;																				
		int candidates =140;																	
		ColorTracker Object = ColorTracker(list_bbox_gt[0],feat,stride,bins,candidates);
		Rect estimated;

		//Main loop for the sequence
		for (;;) {

			//get frame & check if we achieved the end of the video (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);												
			
			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code
			estimated = Object.initiate(frame,frame1);											
			list_bbox_est.push_back(estimated);	
			////////////////////////////////////////////////////////////////////////////////////////////																
			
			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			// show&save data
			imshow("Tracking for "+sequences[s]+" (Green=GT, Red=Estimation)", frame);
			outputvideo.write(frame);//save frame to output video
			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
			
		}

		//similarity between groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);					

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}