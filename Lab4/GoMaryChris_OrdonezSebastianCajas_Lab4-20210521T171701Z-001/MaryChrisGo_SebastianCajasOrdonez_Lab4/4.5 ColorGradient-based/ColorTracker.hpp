#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace colortracker {
	class ColorTracker {
	private:
			int feature;					
			int stride;						
			int bins;						
			int numCandidates;				
			int indx_start, indx_last;		
			float range;					
			Rect model;						
			Rect last_obs;					
			vector<Rect> current_neighbours;
			Mat current_frame;				
			Mat model_hist;					
			vector<float> candidate_scores;	

		public:
			ColorTracker(Rect bbox = Rect(), int feature=5, int stride=2, int bins=16, int numCandidates=121);
			void region();
			Point blob_center(Rect box);
			Rect rect_coord(Point center, int width, int height);
			Mat extract_feature(Rect bbox);
			Mat get_hist(Rect bbox);
			float battacharyya(Mat hist1, Mat hist2);
			void calc_candidates();
			vector<Rect> get_neighbours();
			vector<float> get_norm_scores();

			//Process functions
			Rect initiate(Mat current_frame);
	};
}