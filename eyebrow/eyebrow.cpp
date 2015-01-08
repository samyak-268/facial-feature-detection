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

#include "bgr2hsi.h"
#include "eyebrow_roi.h"

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void help();

string input_image_path;
string face_cascade_path, eye_cascade_path;

Mat_<Vec3b> equalizeImage(const Mat& image_BGR);

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
    Mat image_BGR = imread(input_image_path);
    Mat_<Vec3b> eq_image_BGR = equalizeImage(image_BGR);

    imshow("Original-Image", image_BGR);
    imshow("Equalized-Image", eq_image_BGR);
    
    // Detect faces and eyebrows in image
    // EyebrowROI eyebrow_detector(image_BGR, face_cascade_path, eye_cascade_path);
    // eyebrow_detector.detectEyebrows();
    // eyebrow_detector.displayROI();

    waitKey(0);
    return 0;
}

Mat_<Vec3b> equalizeImage(const Mat& image_BGR)
{
    // Convert image to HSI and extract intensity plane
    BGR2HSI converter(image_BGR);
    Mat image_HSI = converter.convert();
    Mat intensity_plane = converter.extractIntensityPlane(image_HSI);
    
    // Perform histogram equalization on the intensity plane and merge it back
    Mat eq_intensity_plane;
    equalizeHist(intensity_plane, eq_intensity_plane);
    Mat_<Vec3d> equalized_hsi = converter.combinePlanes(converter.returnTrueHSI(), eq_intensity_plane);

    // Convert the HSI image back to BGR
    Mat_<Vec3b> eq_image_BGR = converter.invert(equalized_hsi);
    return eq_image_BGR;
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

