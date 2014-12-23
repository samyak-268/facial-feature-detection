#ifndef _BGR2HSI_CPP
#define _BGR2HSI_CPP

#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "bgr2hsi.h"

using namespace std;
using namespace cv;

#define PI 3.1416

BGR2HSI::BGR2HSI(const Mat& _bgr)
    :bgr(_bgr)
{
    true_hsi.create(bgr.size());
}

BGR2HSI::BGR2HSI(const BGR2HSI& _obj)
{
    bgr = _obj.bgr;
    true_hsi = _obj.true_hsi;
}

int BGR2HSI::Min(int x, int y, int z)
{
    if ((x <= y) && (x <= z))
        return x;
    else if ((y < x) && (y < z))
        return y;
    else
        return z;
}

Mat BGR2HSI::convert()
{
    Mat hsi(bgr.size(), CV_8UC3);
    Mat_<Vec3b> _bgr = bgr;
    Mat_<Vec3b> _hsi = hsi;
    for(int i = 0; i < bgr.rows; ++i)
    {
        for(int j = 0; j < bgr.cols; ++j)
        {
            int B = _bgr(i, j)[0];
            int G = _bgr(i, j)[1];
            int R = _bgr(i, j)[2];
            
            int min = Min(R, G, B);
            double numerator = (double)((2*R - G - B)/2);
            double denominator = (sqrt((R-G)*(R-G) + (R-B)*(G-B)));
            
            double I = (double)(B + G + R)/3;
            double S = (I == 0) ? 0 : (double)(I - min)/I;
            double H = (B <= G) ? acos(numerator/denominator) : 
                (2*PI - acos(numerator/denominator));

            _hsi(i, j)[0] = H;  true_hsi(i, j)[0] = H;
            _hsi(i, j)[1] = S;  true_hsi(i, j)[1] = S;
            _hsi(i, j)[2] = I;  true_hsi(i, j)[2] = I;
        }
    }
    hsi = _hsi;
    return hsi;
}

Mat_<Vec3d> BGR2HSI::returnTrueHSI()
{
    return true_hsi;
}

Mat BGR2HSI::extractIntensityPlane(const Mat_<Vec3b>& hsi)
{
    Mat intensity_plane(hsi.size(), CV_8UC1);
    for(int i = 0; i < hsi.rows; ++i)
    {
        uchar* _intensity_plane_row = intensity_plane.ptr<uchar>(i);
        for(int j = 0; j < hsi.cols; ++j)
            _intensity_plane_row[j] = hsi(i, j)[2];
    }
    return intensity_plane;
}

vector<double> BGR2HSI::calculateHistogram(const Mat& intensity_plane)
{
    vector<int> histogram_frequency(256, 0);
    vector<double> histogram(256, 0.0);
    int total_pixels = (intensity_plane.rows * intensity_plane.cols);

    for(int i = 0; i < intensity_plane.rows; ++i)
    {
        const uchar* intensity_plane_row = intensity_plane.ptr<uchar>(i);
        for(int j = 0; j < intensity_plane.cols; ++j)
            histogram_frequency[intensity_plane_row[j]] += 1;
    }

    for(int i = 0; i < 256; ++i)
        histogram[i] = (double)histogram_frequency[i]/total_pixels;

    return histogram;
}

vector<double> BGR2HSI::equalizeHistogram(const vector<double>& histogram)
{
    int L = histogram.size();
    vector<double> equalized_histogram(L, 0.0);

    double cumulative_frequency_sum = 0.0;
    for(int i = 0; i < L; ++i)
    {
        cumulative_frequency_sum += histogram[i];
        equalized_histogram[i] = (cumulative_frequency_sum * (L-1));
    }
    
    return equalized_histogram;
}

#endif
