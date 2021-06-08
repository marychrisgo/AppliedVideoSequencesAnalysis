#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace tracking {
	class GradientTracker {

	private:
		int stride;						
		int bins;						
		int candidates;				
		int indx_start, indx_last;		
	public:
		Mat present;				// current frame
		vector<float> model_feat;		// for storing HOG features
		HOGDescriptor HoG_d;		
		vector<float> scores;	
		void candidate_region();
		Point blob_center(Rect box);
		Rect rect_coord(Point center, int width, int height);
		vector<float> get_gradient(Rect bounding_box);
		double l2distance(vector<float> hist1, vector<float> hist2);
		void calcCandidates_G();
		GradientTracker(Rect bounding_box, int stride=2, int bins=8, int candidates=121);
		Rect initiate(Mat present, bool& first_frame);
		Rect model;						
		Rect last_obs;					
		Rect current_box;				
		vector<Rect> neighbours;
	};
}