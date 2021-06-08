#include <opencv2/opencv.hpp>
#include "ColorTracker.hpp"
#include "GradientTracker.hpp"
using namespace std;
using namespace cv;
using namespace colortracker;
using namespace gradtracker;

namespace tracking {
	class FusedTracker {
	private:
		char mode;							
		vector <float> combined_scores;		
		bool first_frame;					
		Mat current_frame;					
		Rect object;						
		vector<Rect> neighbours;			
		ColorTracker ctSetting;			
		GradientTracker gtSetting;			
	public:
		FusedTracker(char mode = 'C',Rect bounding_box=Rect(), int feat = 5, int stride = 2, int bins = 16, int candidates = 121);		
		Rect initiate(Mat frame);		
		Rect fusion_results();
	};
}