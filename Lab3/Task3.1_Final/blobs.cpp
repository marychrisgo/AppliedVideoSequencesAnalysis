/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

#include "blobs.hpp"

/**
 *	Draws blobs with different rectangles on the image 'frame'. All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param pBlobList List to store the blobs found
 * \param labelled - true write label and color bb, false does not wirite label nor color bb
 *
 * \return Image containing the draw blobs. If no blobs have to be painted
 *  or arguments are wrong, the function returns a copy of the original "frame".
 *
 */
using namespace cv;
using namespace std;



 Mat paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist, bool labelled)
{
	cv::Mat blobImage;
	//check input conditions and return original if any is not satisfied
	//...
	frame.copyTo(blobImage);
	//required variables to paint
	//...
	//paint each blob of the list
	for(int i = 0; i < bloblist.size(); i++)
	{
		cvBlob blob = bloblist[i]; //get ith blob
		//...
		Scalar color;
		std::string label="";
		switch(blob.label){
		case PERSON:
			color = Scalar(255,0,0); // RED
			label="PERSON";
			break;
		case CAR:
					color = Scalar(0,255,0); // GREEN
					label="CAR";
					break;
		case OBJECT:
					color = Scalar(0,0,255); // BLUE
					label="OBJECT";
					break;
		default:
			color = Scalar(255, 255, 255); // UNKNOWN
			label="UNKOWN";
		}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x+blob.w, blob.y+blob.h);
		// Rectangle: Draws a simple, thick, or filled up-right rectangle using p1 and p2, two opossite corners
		//		  input, 	p1, p2,  color, thickness, line type, shift
		rectangle(blobImage, p1, p2, color, 1, 8, 0); 
		if (labelled)
			{
			rectangle(blobImage, p1, p2, color, 1, 8, 0);
			putText(blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
			}
			else
				rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);
	}

	//destroy all resources (if required)
	//...

	//return the image to show
	return blobImage;
}


/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image) 
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation) 
 */

int extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist, int connectivity)
{	
	//check input conditions and return -1 if any is not satisfied
	
    if((connectivity!=8) & (connectivity!=4))
	{
		cout<<"Bad connectivity input!!"<<endl;
		return -1;
	}

	Mat Output; // image to be updated each time a blob is detected (blob cleared)
	//fgmask.convertTo(Output,CV_32SC1);
	fgmask.copyTo(Output);
	
	//clear blob list (to fill with this function)
	bloblist.clear();
			
	// void creation of a unqie blob in the center
	cvBlob blob;
	//blob=initBlob(1, fgmask.cols/2, fgmask.rows/2, fgmask.cols/4, fgmask.rows/4);
	//bloblist.push_back(blob);

	int IDcount = 0;

	for(int i=0; i< Output.rows;++i){
		for(int j=0; j<Output.cols;++j){
				if(Output.at<uchar>(i,j)==255){
				        Rect Selection;
					IDcount +=1;
					// Floodfill reune las imagenes completas, documentacion dice que: seedpoint -> starting point
					// Entonces Point detecta un punto del fgmask u floodFill le coloca el valor de IDcount
					cv::floodFill(Output,Point(j,i),IDcount, &Selection,0,0,connectivity);
					// Add ID+BLOB to blob
					blob=initBlob(IDcount, Selection.x, Selection.y, Selection.width, Selection.height);
					// Store BLOB on bloblist
					bloblist.push_back(blob);			
				}
		}
	}
	//return OK code
	return 1;
}


int removeSmallBlobs(std::vector<cvBlob> bloblist_in, std::vector<cvBlob> &bloblist_out, int min_width, int min_height)
{
	//check input conditions and return -1 if any is not satisfied
	//clear blob list (to fill with this function)
	bloblist_out.clear();

	for(int i = 0; i < bloblist_in.size(); i++)
	{
		cvBlob blob_in = bloblist_in[i]; //get ith blob

		if ((blob_in.w > min_width) | (blob_in.h > min_height )){
		bloblist_out.push_back(blob_in); // void implementation (does not remove)
		}
	}
	//destroy all resources
	//...
	
	//return OK code
	return 1;
}
 
 int biggest_blob(std::vector<cvBlob> &bloblist) 
 	{
 	int biggest_blob = 0;
 	for(int i = 0; i < bloblist.size(); i++)
 	        {
 		cvBlob blob = bloblist[i]; //get i-th blob
 		if ((blob.w*blob.h > biggest_blob)){
		         biggest_blob = i;			                        
		}
 		
 	}                        
 	return biggest_blob;
 
 }



