#include <opencv2/opencv.hpp>
#include "GradientTracker.hpp"

using namespace gradtracker;
using namespace std;
using namespace cv;


/**
 * Candidates (current detection) are generated from NxN neighborhood with 
 * the center of the previous detection,for calculating candidate generation neighbourhood.
 * For this task, we obtain the square root of the given number canditates and
 * use the stride size to instantiate the starting and ending indexes, which will
 * determine the research area for candidates selection
 * **/
void GradientTracker::calc_bounds_candidates() {
	int N = sqrt(numCandidates);
	indx_start = (N / 2) * stride;
	if (N % 2 == 0) {	
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
Point GradientTracker::blob_center(Rect box) {
	float x = box.x + box.width * 0.5;
	float y = box.y + box.height * 0.5;
	return Point(x, y);
}

/**
 * It extracts the (xmin, ymin, width, height), given by the bounding box on a Point-type variable
 * */
Rect GradientTracker::rect_coord(Point center, int width, int height) {
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
void GradientTracker::region() {
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
GradientTracker::GradientTracker(Rect bbox, int stride, int bins, int numCandidates)
{
	model = bbox;								
	last_obs = bbox;							
	this->stride = stride;						
	this->bins = bins;							
	this->numCandidates = numCandidates;		
	calc_bounds_candidates();			
	descriptor = HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), bins, 1);
}

/** 
 * It uses the given bounding box to extract the HOG descriptor by first, converting the
 * Rect type rectangle coordinates into a Mat type, then resizes the image and ultimately
 * computes the HOG descriptor on a feature vector-
 * @param bounding_box: Rect coordinates of area to extract feature
 * */
// https://answers.opencv.org/question/70491/matching-hog-images-with-opencv-in-c/
// using HOGdescriptor
vector<float> GradientTracker::get_gradient(Rect bbox) {
	Mat cropImg;
	current_frame(bbox).copyTo(cropImg);
	resize(cropImg, cropImg, Size(64, 128));
	vector<float> descs;	
	descriptor.compute(cropImg, descs, Size(8, 8), Size(0, 0));
	return descs;
}

/**
 * It applies the Euclidean distance between two different vectors, in this case, it computes the
 * Euclidean distance or Squared sum between two different descriptors features
 * @param hist1: model
 * @param hist2: candidate
 * */
double GradientTracker::l2distance(vector<float> hist1, vector<float> hist2) {
	return norm(hist1, hist2, NORM_L2);
}

// candidate regions in the neighbourhood
vector<Rect> GradientTracker::get_neighbours() {
	return current_neighbours;
}

// normalised L1 scores
vector<float> GradientTracker::get_norm_scores() {
	normalize(candidate_scores, candidate_scores, 1.0, 0.0, NORM_L1);
	return candidate_scores;
}

/** 
 * The function inititiate receives the current frame and a boolean variable
 * which will determine whether the received frame corresponds or not to the first
 * frame. If this is the case, then the flag value is changed to false and the 
 * model histogram is obtained, using the function get_hist. Then, using the neighboring
 * indexes calculated previously, the L2 distance is calculated for each candidates
 * model and ultimately, the candidate with most similarity is selected as the best candidate. As
 * final outcome, the results are plotted on a new window. 
 * 
 * @param present: Given frame
 * @param firt_frame: boolean variable 
 * **/
Rect GradientTracker::initiate(Mat current_frame) {

	cvtColor(current_frame, this->current_frame, cv::COLOR_RGB2GRAY);
	if (model_hist==vector<float>()) {
		model_hist = get_gradient(model);
	}
	// clearing list
	current_neighbours.clear();
	region();
	candidate_scores.clear();
	for (unsigned int i = 0; i < current_neighbours.size(); i++) {
		candidate_scores.push_back(l2distance(model_hist, get_gradient(current_neighbours[i])));
	}
	int minElem = min_element(candidate_scores.begin(), candidate_scores.end()) - candidate_scores.begin();
	last_obs = current_neighbours[minElem];
	return last_obs;
}


