#ifndef _EYEBROW_ROI_CPP
#define _EYEBROW_ROI_CPP

#include <cmath>
#include "eyebrow_roi.h"
using namespace std;
using namespace cv;

EyebrowROI::EyebrowROI(const Mat& _image, const string& _face_cascade_path, 
        const string& _eye_cascade_path)
    :image(_image), face_cascade_path(_face_cascade_path), eye_cascade_path(_eye_cascade_path)
{
    face_cascade.load(face_cascade_path);
    eye_cascade.load(eye_cascade_path);
}

EyebrowROI::EyebrowROI(const EyebrowROI& _obj)
{
    image = _obj.image;
    face_cascade_path = _obj.face_cascade_path;
    eye_cascade_path = _obj.eye_cascade_path;
    face_cascade = _obj.face_cascade;
    eye_cascade = _obj.eye_cascade;
}

void EyebrowROI::detectFace()
{
    face_cascade.detectMultiScale(image, faces, 1.15, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    return;
}

void EyebrowROI::detectEyebrows()
{
    detectFace();
    for(unsigned int i = 0; i < faces.size(); ++i)
    {
        Rect_<int> face = faces[i];
        face_roi = image(Rect(face.x, face.y, face.width, face.height));

        eye_cascade.detectMultiScale(face_roi, eyes, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    }
    return;
}

vector<Mat> EyebrowROI::displayROI()
{
    for(unsigned int i = 0; i < faces.size(); ++i)
    {
        Rect_<int> face = faces[i];
        /*
        rectangle(image, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
                Scalar(0, 0, 255), 1, 4);
        */

        for(unsigned int j = 0; j < eyes.size(); ++j)
        {
            Rect_<int> e = eyes[j];
            
            // Calculate parameters for eyebrow bounding box from those of eye bounding box
            int eyebrow_bbox_x = e.x;
            int eyebrow_bbox_y = (e.y - e.height/5);
            
            int eyebrow_bbox_height = (e.height * 3)/5;
            int eyebrow_bbox_width = round((double)e.width * 1.6);
            
            // Save and mark eyebrow region
            eyebrows_roi.push_back( face_roi(Rect(eyebrow_bbox_x, eyebrow_bbox_y, 
                            eyebrow_bbox_width, eyebrow_bbox_height)) );

            /*
            rectangle(face_roi, Point(eyebrow_bbox_x, eyebrow_bbox_y), 
                    Point(eyebrow_bbox_x+eyebrow_bbox_width, eyebrow_bbox_y+eyebrow_bbox_height), 
                    Scalar(255, 0, 0), 1, 4);
            */
        }
    }
    // imshow("Eyebrow_Detection", image);
    return eyebrows_roi;
}

#endif
