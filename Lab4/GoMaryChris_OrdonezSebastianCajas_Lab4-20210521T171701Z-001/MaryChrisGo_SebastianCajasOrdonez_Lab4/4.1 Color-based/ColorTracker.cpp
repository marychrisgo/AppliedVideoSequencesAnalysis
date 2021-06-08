#include <opencv2/opencv.hpp>
#include "ColorTracker.hpp"
#include "ShowManyImages.hpp"

using namespace tracking;
using namespace std;
using namespace cv;

/**
 * Candidates (current detection) are generated from NxN neighborhood with 
 * the center of the previous detection,for calculating candidate generation neighbourhood.
 * For this task, we obtain the square root of the given number canditates and
 * use the stride size to instantiate the starting and ending indexes, which will
 * determine the research area for candidates selection
 * **/

void ColorTracker::calcCandidates() {
	int N = sqrt(candidates);
	indx_start = (N / 2) * stride; // will provide the search area for the next candidate
	indx_last = ((N / 2) - 1) * stride;
	}
	
/**
 * It obtains the blob center of the bounding box.
 * @param box: bounding box - rect type
 * */
Point ColorTracker::blob_center(Rect box) {
	float x = box.x + box.width * 0.5;
	float y = box.y + box.height * 0.5;
	return Point(x, y);
}


/**
 * It extracts the (xmin, ymin, width, height), given by the bounding box on a Point-type variable
 * */
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
//To generate list of candidate regions
void ColorTracker::region() {
	Point center = blob_center(last_obs);
	for (int i = - indx_start; i <=indx_last; i += stride) {
		for (int j = - indx_start; j <= indx_last; j += stride) {
			neighbours.push_back(rect_coord(Point(center.x + i, center.y + j), last_obs.width, last_obs.height));
		}
	}
}

/**
 * Constructor - initializing tracker parameters with initial values. On this first stage,
 * frame 1 is used to calculate the model reference over all sequences.
 * */
ColorTracker::ColorTracker(Rect bounding_box, int feat, int stride, int bins, int candidates)
{
	this->feat = feat;					
	this->stride = stride;						
	this->bins = bins;							
	this->candidates = candidates;		
	calcCandidates();
    model = bounding_box;								
	last_obs = bounding_box;												
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
//Returns the candidate region in appropriate feature space
Mat  ColorTracker::extract_feature(Rect bounding_box){
	Mat tracked_img, orig_img;
	present(bounding_box).copyTo(tracked_img);
	range = 256;												
	if (feat == 1) {
		cvtColor(tracked_img, orig_img, COLOR_BGR2GRAY); //grayscale
	}
	else if (feat > 1 && feat <= 4) {
		Mat bgr[3]; // rgb channel 
		split(tracked_img, bgr);
		switch (feat) {
		case(2): {
			bgr[0].copyTo(orig_img); 		// blue		
			break;
		}
		case(3): {
			bgr[1].copyTo(orig_img); 		// green		
			break;
		}
		case(4): {
			bgr[2].copyTo(orig_img); 		// red		
			break;
		}
		default:
			break;
		}
	}
	//HSV Channel
	else if (feat > 4 && feat <= 7) {
		Mat  HSVImg, hsv[3];
		cvtColor(tracked_img, HSVImg, COLOR_BGR2HSV);
		split(HSVImg, hsv);
		switch (feat) {
		case(5): {
			range = 180;										
			hsv[0].copyTo(orig_img); 		// hue		
			break;
		}
		case(6): {
			hsv[1].copyTo(orig_img);		// saturation			
			break;
		}
		case(7): {
			hsv[2].copyTo(orig_img);		// value 	
			break;
		}
		default:
			break;
		}
	}
	else {
		cvtColor(tracked_img, orig_img, COLOR_BGR2GRAY); 		// convert to grayscale	
	}
	return orig_img;

}


/** 
 * get_hist uses the function extract_Feature to extract the feature color from the given bounding box.
 * The window containing the rectangular area for the tracked object is converted to a mat instance, to
 * ultimately being able to quantize the selected bins through a histogram. The final outcome is normalized
 * between 0 and 1 using the L1 normalization or manhathan distance. 
 * @param boundingbox: refers to the rectangular shaped instance to contain a desired tracked object. It can either
 * be the model reference (ground-truth) or as well any of the given candidates.
 **/
Mat ColorTracker::get_hist(Rect bounding_box) {
	Mat orig_img = extract_feature(bounding_box);
	float range_hist[] = {0, range}; 									
	const float* histRange = {range_hist};

    Mat gen_hist;
	calcHist(&orig_img, 1, 0, Mat(), gen_hist, 1, &bins, &histRange);		
	normalize(gen_hist, gen_hist, 1, 0, NORM_L1, -1, Mat());					
	return gen_hist;
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
Rect ColorTracker::initiate(Mat present, bool& first_frame) {
	present.copyTo(this->present);
	if (first_frame) {
		first_frame = false;
		model_hist = get_hist(model);
	}

	// clearing the previous list
	neighbours.clear();
	region();
	score.clear();

    // looping  it in a list		
	for (unsigned int i = 0; i < neighbours.size(); i++) {
		score.push_back(battacharyya(model_hist, get_hist(neighbours[i]))); 
	}

	// minimum  battacharyya distance 
	int min_distance = min_element(score.begin(), score.end()) - score.begin();
	last_obs = neighbours[min_distance];
    
    // Show the images
	int hist_width = 500; 
    int hist_height = 400;									
	int bin_width = cvRound( (double) hist_width/bins );						
	Mat histImage_mod(hist_height, hist_width, CV_8UC3, Scalar( 0,0,0) );		
	Mat histImage_can(hist_height, hist_width, CV_8UC3, Scalar( 0,0,0) );		
	Mat final_hist_can = get_hist(last_obs); //candidate
	final_hist_can = histImage_can.rows*final_hist_can; // normalized histogram

	for( int i = 1; i < bins; i++ )
	{
	    line( histImage_can, Point( bin_width*(i-1), hist_height - cvRound(final_hist_can.at<float>(i-1))), Point( bin_width*(i), 
            hist_height - cvRound(final_hist_can.at<float>(i)) ), Scalar( 0,255, 0), 2, 8, 0  );
	}

	Mat fin_cand; // final candidate
	present(last_obs).copyTo(fin_cand);
	Mat feat_cand; // feature
	extract_feature(last_obs).copyTo(feat_cand);
    rectangle(present, last_obs, Scalar(0, 0, 255));
	ShowManyImages("Tracking", 4, present,fin_cand, histImage_can, feat_cand, histImage_mod);
	return last_obs;
}

