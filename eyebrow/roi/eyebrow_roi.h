#ifndef _EYEBROW_ROI_H
#define _EYEBROW_ROI_H

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class EyebrowROI
{
    private:
        Mat image;
        string face_cascade_path;
        string eye_cascade_path;
        CascadeClassifier face_cascade;
        CascadeClassifier eye_cascade;


    public:
        Mat face_roi;
        vector<Rect_<int> > faces;
        vector<Rect_<int> > eyes;
        
        EyebrowROI(const Mat& _image, const string& _face_cascade_path, 
                const string& _eye_cascade_path);
        EyebrowROI(const EyebrowROI& _obj);
        void detectFace();
        void detectEyebrows();
        void displayROI();
};

#endif
