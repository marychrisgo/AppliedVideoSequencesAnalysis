#include <opencv2/opencv.hpp>
#include "ColorTracker.hpp"
#include "ShowManyImages.hpp"

using namespace colortracker;
using namespace std;
using namespace cv;

/**
 * Candidates (current detection) are generated from NxN neighborhood with 
 * the center of the previous detection,for calculating candidate generation neighbourhood.
 * For this task, we obtain the square root of the given number canditates and
 * use the stride size to instantiate the starting and ending indexes, which will
 * determine the research area for candidates selection
 * **/

void ColorTracker::calc_candidates() {
	int N = sqrt(numCandidates);
	indx_start = (N / 2) * stride;
	if (N % 2 == 0 ) {	
		indx_last = ((N / 2) - 1) * stride;
	}
	else {
		indx_last = (N / 2) * stride;
	}
}
/**
 * It obtains the blob center of the bounding box.
 * @param box: bounding box - rect type
 * */
// center of bbox
Point ColorTracker::blob_center(Rect box) {
	float x = box.x + box.width * 0.5;
	float y = box.y + box.height * 0.5;
	return Point(x, y);
}

/**
 * It extracts the (xmin, ymin, width, height), given by the bounding box on a Point-type variable
 * */
// rectangle coordinates
Rect ColorTracker::rect_coord(Point center, int width, int height) {
	float xmin = center.x - width * 0.5;
	float ymin = center.y - height * 0.5;
	return Rect(xmin, ymin, width, height);
}

/**
 * region instantiates the candidates on a vector, using the initial and ending indexes determined by calcCandidates()
 * For doung so, it extracts the x,y coordinates and its correspondent width and height using the function blob_Center()
 * @param last_obs: refers to the bounding box to extract color feature. 
 * @param neighbours: stores the possible neighbors for the given bounding box.
 * */
void ColorTracker::region() {
	Point center = blob_center(last_obs);
	for (int i = center.x - indx_start; i <= center.x + indx_last; i += stride) {
		for (int j = center.y - indx_start; j <= center.y + indx_last; j += stride) {
			Rect rect = rect_coord(Point(i, j), last_obs.width, last_obs.height);
			if(rect.x>0 && rect.y>0 && rect.x+last_obs.width<current_frame.cols&& rect.y+last_obs.height<current_frame.rows)
				current_neighbours.push_back(rect);
		}
	}
}

/**
 * Constructor - initializing tracker parameters with initial values. On this first stage,
 * frame 1 is used to calculate the model reference over all sequences.
 * */
ColorTracker::ColorTracker(Rect bbox, int feature, int stride, int bins, int numCandidates)
{
	model = bbox;								
	last_obs = bbox;							
	this->feature = feature;					
	this->stride = stride;						
	this->bins = bins;							
	this->numCandidates = numCandidates;		
	calc_candidates();					
}

/** 
 * extract_feature receives the selected region - bounding box, to extract the especified feature. First, 
 * the image is converted to gray scale, then, according to the feature parameter, the selected color is extracted.
 * The possible features can either be R,G,B from RGB color model, or H, S, V from HSV model color. Finally, 
 * the feature is returned in Mat format.
 * @param bounding_box: rect instance containing coordinates of area to extract feature
 * @param orig_img: Stores the final feature color.
 * 
 * */
Mat  ColorTracker::extract_feature(Rect bbox){

	Mat cropImg, featImg;
	current_frame(bbox).copyTo(cropImg);
	range = 256;												
	if (feature == 1) {
		cvtColor(cropImg, featImg, COLOR_BGR2GRAY);
	}
	else if (feature > 1 && feature <= 4) {

		Mat bgr_channels[3];
		split(cropImg, bgr_channels);

		switch (feature) {
		case(2): {
			bgr_channels[0].copyTo(featImg); 					//Blue 
			break;
		}
		case(3): {
			bgr_channels[1].copyTo(featImg); 					//Green 
			break;
		}
		case(4): {
			bgr_channels[2].copyTo(featImg); 					//Red 
			break;
		}
		default:
			break;
		}
	}
	//HSV Channel
	else if (feature > 4 && feature <= 7) {
		Mat  HSVImg, hsv_channels[3];
		cvtColor(cropImg, HSVImg, COLOR_BGR2HSV);
		split(HSVImg, hsv_channels);
		switch (feature) {
		case(5): {
			range = 180;										
			hsv_channels[0].copyTo(featImg); 					//Hue 
			break;
		}
		case(6): {
			hsv_channels[1].copyTo(featImg);					//Saturation 
			break;
		}
		case(7): {
			hsv_channels[2].copyTo(featImg);					//Value Chanel
			break;
		}
		default:
			break;
		}
	}
	else {

		cvtColor(cropImg, featImg, COLOR_BGR2GRAY); 			//GrayScale if invalid feature value!
	}
	return featImg;

}

/** 
 * get_hist uses the function extract_Feature to extract the feature color from the given bounding box.
 * The window containing the rectangular area for the tracked object is converted to a mat instance, to
 * ultimately being able to quantize the selected bins through a histogram. The final outcome is normalized
 * between 0 and 1 using the L1 normalization or manhathan distance. 
 * @param boundingbox: refers to the rectangular shaped instance to contain a desired tracked object. It can either
 * be the model reference (ground-truth) or as well any of the given candidates.
 **/
Mat ColorTracker::get_hist(Rect bbox) {
	Mat featImg = extract_feature(bbox);
	Mat hist;
	float range_hist[] = { 0, range }; 									
	const float* histRange = { range_hist };
	calcHist(&featImg, 1, 0, Mat(), hist, 1, &bins, &histRange);		
	normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());					
	return hist;
}


/** 
 * Battacharya Distance is calculated using two different inputs, the model or ground-truth
 * and the candidate. This pricess is performed using the function compareHist  to get a numerical
 *  parameter that express how well two histograms match with each other. The CV_COMP_BHATTACHARYYA input
 * defines the Bhattacharyya distance equation to compute the similarity between both entrances. 
 * @param hist1: model histogram from ground-truth 
 * @param hist2: candidate histogram. 
 * **/ 
float ColorTracker::battacharyya(Mat hist1, Mat hist2) {
	return compareHist(hist1,hist2,HISTCMP_BHATTACHARYYA);
}

// candidate regions to neighbour
vector <Rect> ColorTracker::get_neighbours() {
	return current_neighbours;
}

// normalizing scores
vector<float> ColorTracker::get_norm_scores() {
	normalize(candidate_scores, candidate_scores, 1.0, 0.0, NORM_L1);
	return candidate_scores;
}

/** 
 * The function inititiate receives the current frame and a boolean variable
 * which will determine whether the received frame corresponds or not to the first
 * frame. If this is the case, then the flag value is changed to false and the 
 * model histogram is obtained, using the function get_hist. Then, using the neighboring
 * indexes calculated previously, the battacharyya distance is calculated for each candidates
 * model and ultimately, the candidate with most similarity is selected as the best candidate. As
 * final outcome, the results are plotted on a new window. 
 * 
 * @param present: Given frame
 * @param firt_frame: boolean variable 
 * **/
Rect ColorTracker::initiate(Mat current_frame) {
	current_frame.copyTo(this->current_frame);
	if (model_hist.empty()) {
			model_hist = get_hist(model);
	}
	// clearing list
	current_neighbours.clear();
	region();
	candidate_scores.clear();
	// looping to candidates
	for (unsigned int i = 0; i < current_neighbours.size(); i++) {
		candidate_scores.push_back(battacharyya(model_hist,
					get_hist(current_neighbours[i]))); 		
	}
	int minElem = min_element(candidate_scores.begin(), candidate_scores.end()) - candidate_scores.begin();
	last_obs = current_neighbours[minElem];
}

