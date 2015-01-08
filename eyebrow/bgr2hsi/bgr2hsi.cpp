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

Mat_<Vec3b> BGR2HSI::invert(const Mat_<Vec3d>& hsi)
{
    Mat_<Vec3b> bgr(hsi.size());
    for(int i = 0; i < hsi.rows; ++i)
    {
        for(int j = 0; j < hsi.cols; ++j)
        {
            double H = hsi(i, j)[0];
            double S = hsi(i, j)[1];
            double I = hsi(i, j)[2];

            int H_degrees = round((H * 180) / PI);
            int term1 = (int)(I + (2 * I * S));
            int term2 = (int)(I - (I * S));
            
            double trig = 0.0;
            int R = 0, G = 0, B = 0;
            
            if(H == 0)
            {
                R = term1;
                G = term2;
                B = term2;
            }
            else if((H_degrees > 0) && (H_degrees < 120))
            {
                trig = cos(H) / cos(PI/3 - H);
                
                R = (int)(I + (I*S*trig));
                G = (int)(I + I*S*(1 - trig));
                B = term2;
            }
            else if(H_degrees == 120)
            {
                R = term2;
                G = term1;
                B = term2;
            }
            else if((H_degrees > 120) && (H_degrees < 240))
            {
                trig = cos(H - (2*PI)/3) / cos(PI - H);

                R = term2;
                G = (int)(I + (I*S*trig));
                B = (int)(I + I*S*(1 - trig));
            }
            else if(H_degrees == 240)
            {
                R = term2;
                G = term2;
                B = term1;
            }
            else if((H_degrees > 240) && (H_degrees <= 360))
            {
                trig = cos(H - (4*PI)/3) / cos((5*PI)/3 - H);

                R = (int)(I + I*S*(1 - trig));
                G = term2;
                B = (int)(I + (I*S*trig));
            }

            bgr(i, j)[0] = B;
            bgr(i, j)[1] = G;
            bgr(i, j)[2] = R;
        }
    }
    return bgr;
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

Mat_<Vec3d> BGR2HSI::combinePlanes(const Mat_<Vec3d>& hsi, const Mat& intensity_plane)
{
    Mat_<Vec3d> equalized_hsi(true_hsi.size());
    Mat_<Vec3d> _hsi = hsi;

    for(int i = 0; i < true_hsi.rows; ++i)
    {
        for(int j = 0; j < true_hsi.cols; ++j)
        {
            equalized_hsi(i, j)[0] = _hsi(i, j)[0];
            equalized_hsi(i, j)[1] = _hsi(i, j)[1];
            equalized_hsi(i, j)[2] = intensity_plane.at<uchar>(i, j);
        }
    }
    return equalized_hsi;
}

#endif
