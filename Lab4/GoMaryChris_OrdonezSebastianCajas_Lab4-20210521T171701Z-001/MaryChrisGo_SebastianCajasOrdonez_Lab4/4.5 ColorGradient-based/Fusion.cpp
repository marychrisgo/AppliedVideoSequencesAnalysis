#include <opencv2/opencv.hpp>
#include "Fusion.hpp"
#include "ColorTracker.hpp"
#include "GradientTracker.hpp"
using namespace std;
using namespace cv;
using namespace tracking;
using namespace colortracker;
using namespace gradtracker;

/**
 * Constructor - initializing tracker parameters with initial values. This aims to 
 * fuse two trackers, obtaining the ColorTracker and GracientTracker.
 * */
FusedTracker::FusedTracker(char mode , Rect bounding_box , int feat, int stride, int bins, int candidates) {
	this->mode = mode;
	ctSetting = ColorTracker(bounding_box, feat, stride, bins, candidates);
	gtSetting = GradientTracker(bounding_box, stride, bins, candidates);

}

/**
 * It aims to combine the results of both trackers. It calculates with both the trackers and add the
 * two normalized scores. 
 * */
Rect FusedTracker::fusion_results() {
	Rect frame_temp;
	frame_temp = ctSetting.initiate(current_frame);
	frame_temp = gtSetting.initiate(current_frame);
	neighbours = ctSetting.get_neighbours();

	vector<float> colorscore = ctSetting.get_norm_scores();
	vector<float> gradientscore = gtSetting.get_norm_scores();

	// Add the two scores
	transform(colorscore.begin(), colorscore.end(), gradientscore.begin(), colorscore.begin(), std::plus<float>());

	int minElem = min_element(colorscore.begin(), colorscore.end()) - colorscore.begin();
	return neighbours[minElem];
}

/**
 * It aims to implement a tracker based on the mode parameter. 
 * */
Rect FusedTracker::initiate(Mat frame) {
	frame.copyTo(this->current_frame);
	switch(mode)
	{
	case ('C'): {
		object = ctSetting.initiate(current_frame);
		break;
	}
	case ('G'): {
		object = gtSetting.initiate(current_frame);
		break;
	}
	default: {

		object = fusion_results();
		break;
	}
	}
	return object;
}

