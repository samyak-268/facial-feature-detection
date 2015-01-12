/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 * A program to automatically detect eyebrows and eyebrow key points using a color-based
 * method of eyebrow segmentation that extracts a pseudo-hue plane to separate the eyebrow 
 * from the skin region. 
 *
 * This code is an implementation of the following paper
 * Majumder, A. Singh, M. and Behera, L., "Automatic eyebrow features detection and realization 
 * of avatar for real time eyebrow movement", 7th IEEE International Conference on 
 * Industrial and Information Systems (ICIIS),2012
 */

#include "eyebrow_roi.h"

#include <iostream>
#include <limits>
#include <cmath>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void help();

string input_image_path;
string face_cascade_path, eye_cascade_path;

double getMax(const Mat_<double>& intensity_plane);
double getMin(const Mat_<double>& intensity_plane);

Mat_<uchar> extractPseudoHue(const Mat_<Vec3b> image_BGR);

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        help();
        return 1;
    }

    input_image_path = argv[1];
    face_cascade_path = argv[2];
    eye_cascade_path = argv[3];

    // Load and equalize image
    Mat_<Vec3b> image_BGR = imread(input_image_path);
    // Mat_<Vec3b> eq_image_BGR = equalizeImage(image_BGR);
    
    vector<Mat> channels;
    Mat img_hist_equalized;
    cvtColor(image_BGR, img_hist_equalized, CV_BGR2YCrCb);
    split(img_hist_equalized, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, img_hist_equalized);
    cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR);

    // Extract pseudo-hue plane
    Mat_<uchar> normalized_pseudo_hue_plane = extractPseudoHue(img_hist_equalized);

    imshow("Original-Image", image_BGR);
    imshow("Pseudo-Hue-Plane", normalized_pseudo_hue_plane);
    
    // Detect faces and eyebrows in image
    // EyebrowROI eyebrow_detector(image_BGR, face_cascade_path, eye_cascade_path);
    // eyebrow_detector.detectEyebrows();
    // eyebrow_detector.displayROI();

    waitKey(0);
    return 0;
}

Mat_<uchar> extractPseudoHue(const Mat_<Vec3b> image_BGR)
{
    Mat_<double> pseudo_hue_plane(image_BGR.size());
    
    // Calculate the pseudo-hue plane
    for(int i = 0; i < image_BGR.rows; ++i)
    {
        for(int j = 0; j < image_BGR.cols; ++j)
        {
            int B = image_BGR(i, j)[0];
            int G = image_BGR(i, j)[1];
            int R = image_BGR(i, j)[2];

            if((R == 0) && (G == 0))
                pseudo_hue_plane.at<double>(i, j) = 0.0;
            else
                pseudo_hue_plane.at<double>(i, j) = ((double)R / (R+G));
        }
    }
    
    // Normalize the pseudo-hue plane
    double Hmin = getMin(pseudo_hue_plane);
    double Hmax = getMax(pseudo_hue_plane);
    double span = (Hmax - Hmin);

    double Hnorm = 0.0;
    Mat_<uchar> normalized_pseudo_hue_plane(pseudo_hue_plane.size());
    for(int i = 0; i < pseudo_hue_plane.rows; ++i)
    {
        for(int j = 0; j < pseudo_hue_plane.cols; ++j)
        {
            Hnorm= ((pseudo_hue_plane.at<double>(i, j) - Hmin) / span);
            normalized_pseudo_hue_plane.at<uchar>(i, j) = round(Hnorm * 255);
        }
    }

    return normalized_pseudo_hue_plane;
}

double getMax(const Mat_<double>& intensity_plane)
{
    double max_entry = intensity_plane.at<double>(0, 0);
    for(int i = 0; i < intensity_plane.rows; ++i)
    {
        for(int j = 0; j < intensity_plane.cols; ++j)
        {
            if(intensity_plane.at<double>(i, j) >= max_entry + numeric_limits<double>::epsilon())
                max_entry = intensity_plane.at<double>(i, j);
        }
    }
    return max_entry;
}

double getMin(const Mat_<double>& intensity_plane)
{
    double min_entry = intensity_plane.at<double>(0, 0);
    for(int i = 0; i < intensity_plane.rows; ++i)
    {
        for(int j = 0; j < intensity_plane.cols; ++j)
        {
            if(intensity_plane.at<double>(i, j) <= min_entry + numeric_limits<double>::epsilon())
                min_entry = intensity_plane.at<double>(i, j);
        }
    }
    return min_entry;
}

void help()
{
    cout << "\nThis program demonstrates eyebrow and eyebrow key-point detection using a color-based\n"
        "method of eyebrow segmentation that extracts a hue-plane to separate\n" 
        "the eyebrow from the skin region.\n";

    cout << "\nUSAGE: ./eyebrow [IMAGE] [FACE_CASCADE] [EYE_CASCADE]\n"
        "IMAGE\n\tPath to the image of a face taken as input.\n"
        "FACE_CASCSDE\n\t Path to a haarcascade classifier for face detection.\n"
        "EYE_CASCSDE\n\t Path to a haarcascade classifier for eye detection.\n";

}

