#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace gradtracker {
	class GradientTracker {
	private:
		int stride;					
		int bins;					
		int numCandidates;				
		int indx_start, indx_last;
		
	public:
		Rect model;						
		Rect last_obs;					
		vector<Rect> current_neighbours;
		Mat current_frame;				
		vector<float> model_hist;		
		HOGDescriptor descriptor;		
		vector<float> candidate_scores;	
		GradientTracker(Rect bbox = Rect(), int stride=2, int bins=9, int numCandidates=100);
		void region();
		Point blob_center(Rect box);
		Rect rect_coord(Point center, int width, int height);
		vector<float> get_gradient(Rect bbox);
		double l2distance(vector<float> hist1, vector<float> hist2);
		void calc_bounds_candidates();
		vector<Rect> get_neighbours();
		vector<float> get_norm_scores();
		Rect initiate(Mat current_frame);
	};
}