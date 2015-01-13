/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 * This code is an implementation of the following paper
 * Juliano L. Moreira, Adriana Braun, Soraia R Musse, "Eyes and Eyebrow Detection for 
 * Performance Driven Animation", 23rd SIBGRAPI Conference on Graphics, Patterns and Images, 2010
 *
 */

#include "eyebrow_roi.h"

#include <iostream>
#include <utility>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

string input_image_path;
string face_cascade_path, eye_cascade_path;

Mat_<uchar> CRTransform(const Mat& image); 
Mat_<uchar> exponentialTransform(const Mat_<uchar>& image);
pair<double, double> returnImageStats(const Mat_<uchar>& image);
Mat_<uchar> binaryThresholding(const Mat_<uchar>& image, const pair<double, double>& stats);

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cout << "Parameters missing!\n";
        return 1;
    }

    input_image_path = argv[1];
    face_cascade_path = argv[2];
    eye_cascade_path = argv[3];

    Mat_<Vec3b> image_BGR = imread(input_image_path);

    // Detect faces and eyebrows in image
    EyebrowROI eyebrow_detector(image_BGR, face_cascade_path, eye_cascade_path);
    eyebrow_detector.detectEyebrows();
    vector<Mat> eyebrows_roi = eyebrow_detector.displayROI();

    // Mat_<uchar> image_exp = exponentialTransform(CRTransform(image_BGR));
    Mat_<uchar> image_exp = exponentialTransform(CRTransform(eyebrows_roi[0]));
    Mat_<uchar> image_binary = binaryThresholding(image_exp, returnImageStats(image_exp));

    vector<vector<Point> > contours;
    findContours(image_binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    imshow("Exponential-Transform", image_binary);


    waitKey(0);
    return 0;
}

Mat_<uchar> CRTransform(const Mat& image)
{
    Mat_<Vec3b> _image = image;
    Mat_<uchar> CR_image(image.size());
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
            CR_image.at<uchar>(i, j) = (255 - _image(i, j)[2]);
    }
    return CR_image;
}

Mat_<uchar> exponentialTransform(const Mat_<uchar>& image)
{
    vector<int> exponential_transform(256, 0);
    for(int i = 0; i < 256; ++i)
        exponential_transform[i] = round(exp((i * log(255)) / 255));

    Mat_<uchar> image_exp(image.size());
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
            image_exp.at<uchar>(i, j) = exponential_transform[image.at<uchar>(i, j)];
    }
    return image_exp;
}

pair<double, double> returnImageStats(const Mat_<uchar>& image)
{
    double mean = 0.0, std_dev = 0.0;
    int total_pixels = (image.rows * image.cols);
    
    int intensity_sum = 0;
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
            intensity_sum += image.at<uchar>(i, j);
    }
    mean = (double)intensity_sum/total_pixels;

    int sum_sq = 0;
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
            sum_sq += ( (image.at<uchar>(i, j) - mean) * (image.at<uchar>(i, j) - mean) );
    }
    std_dev = sqrt((double)sum_sq/total_pixels);

    return make_pair(mean, std_dev);
}

Mat_<uchar> binaryThresholding(const Mat_<uchar>& image, const pair<double, double>& stats)
{
    Mat_<uchar> image_binary(image.size());
    
    double Z = 0.9;
    double threshold = stats.first + (Z * stats.second); 
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
        {
            if(image.at<uchar>(i, j) >= threshold + numeric_limits<double>::epsilon())
                image_binary.at<uchar>(i, j) = 255;
            else
                image_binary.at<uchar>(i, j) = 0;
        }
    }
    return image_binary;
}
