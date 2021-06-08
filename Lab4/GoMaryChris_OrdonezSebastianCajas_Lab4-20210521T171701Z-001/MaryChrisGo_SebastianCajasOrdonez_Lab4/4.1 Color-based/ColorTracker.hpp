#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace tracking {
	class ColorTracker {
	private:
		int feat;					
		int stride;						
		int bins;						
		int candidates;				
		int indx_start, indx_last;		
		float range;					
	public:
        Rect model;					// model candidate
		Rect last_obs;					// est candidate
		Rect current_box;				// current candidate
		vector<Rect> neighbours;
		Mat present;				
		Mat model_hist;					//histogram of the model
		vector<float> score;		// battacharyya distance
		Rect initiate(Mat present, bool& frame1); // for starting 
		ColorTracker(Rect bounding_box, int feat=5, int stride=2, int bins=16, int candidates=121);
		Point blob_center(Rect box);
		Rect rect_coord(Point center, int width, int height);
        void region();
		Mat extract_feature(Rect bounding_box);
		Mat get_hist(Rect bounding_box); 
		float battacharyya(Mat hist1, Mat hist2);
		void calcCandidates();
	};
}