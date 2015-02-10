/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 *
 */

#include <iostream>
#include <cmath>
#include <cfloat>
#include <climits>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Mat_<Vec3b> equalizeImage(Mat_<Vec3b> image_BGR);
Mat_<uchar> transformPseudoHue(Mat_<Vec3b> image);
pair<double, double> Stats(Mat_<double> pseudo_hue_plane);
Mat_<uchar> transformCIELAB(Mat_<Vec3b> image_BGR);
Mat_<uchar> transformLUX(Mat_<Vec3b> image_BGR);
Mat_<uchar> transformModifiedLUX(Mat_<Vec3b> image_BGR);

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "Paramters missing\n";
        return -1;
    }
    
    const string input_image_path = argv[1];
    Mat_<Vec3b> image_BGR = imread(input_image_path);
    Mat_<uchar> image_BGR_eq = equalizeImage(image_BGR);
    
    Mat_<uchar> pseudo_hue_plane = transformPseudoHue(image_BGR);
    // Mat_<uchar> chrominance_plane = transformLUX(image_BGR);
    Mat_<uchar> modified_chrominance_plane = transformModifiedLUX(image_BGR);

    // imshow("Original-Image", image_BGR);
    imshow("Pseudo-Hue", pseudo_hue_plane);
    // imshow("Chrominance-Plane", chrominance_plane);
    imshow("Modified-Chrominance-Plane", modified_chrominance_plane);

    waitKey(0);
    return 0;
}

/*
 * Equalize a BGR image by converting it to the YCrCb space and
 * performing histogram equalization on the Y-plane before
 * merging back
 */
Mat_<Vec3b> equalizeImage(Mat_<Vec3b> image_BGR)
{
    vector<Mat> channels;
    Mat_<Vec3b> image_eq;

    cvtColor(image_BGR, image_eq, CV_BGR2YCrCb);
    split(image_eq, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, image_eq);
    cvtColor(image_eq, image_eq, CV_YCrCb2BGR);

    return image_eq;
}

// Extract the pseudo-hue plane
Mat_<uchar> transformPseudoHue(Mat_<Vec3b> image)
{
    Mat_<double> pseudo_hue(image.size());
    Mat_<uchar> pseudo_hue_norm(image.size());

    int B = 0, G = 0, R = 0;
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
        {
            B = image(i, j)[0];
            G = image(i, j)[1];
            R = image(i, j)[2];

            if(R == 0)
                pseudo_hue.at<double>(i, j) = 0.0;
            else
                pseudo_hue.at<double>(i, j) = (double)R / (R + G);
        }
    }
    pair<double, double> statistics = Stats(pseudo_hue);
    double Hmin = statistics.first;
    double Hmax = statistics.second;
    double Hrange = (Hmax - Hmin);
    
    for(int i = 0; i < image.rows; ++i)
    {
        for(int j = 0; j < image.cols; ++j)
        {
            double  temp = (pseudo_hue.at<double>(i, j) - Hmin) / Hrange;
            pseudo_hue_norm.at<uchar>(i, j) = round(temp * 255);
        }
    }

    return pseudo_hue_norm;
}

pair<double, double> Stats(Mat_<double> pseudo_hue_plane)
{
    double Hmax = DBL_MIN;
    double Hmin = DBL_MAX;

    for(int i = 0; i < pseudo_hue_plane.rows; ++i)
    {
        for(int j = 0; j < pseudo_hue_plane.cols; ++j)
        {
            if(pseudo_hue_plane.at<double>(i, j) >= Hmax + numeric_limits<double>::epsilon())
                Hmax = pseudo_hue_plane.at<double>(i, j);
            if(pseudo_hue_plane.at<double>(i, j) <= Hmin + numeric_limits<double>::epsilon())
                Hmin = pseudo_hue_plane.at<double>(i, j);
        }
    }
    return make_pair(Hmin, Hmax);
}

// CIELAB transformation and using the A-channel
Mat_<uchar> transformCIELAB(Mat_<Vec3b> image_BGR)
{
    Mat_<Vec3b> image_Lab;
    vector<Mat> channels;
    cvtColor(image_BGR, image_Lab, CV_BGR2Lab);
    split(image_Lab, channels);

    Mat_<uchar> image_a = channels[1];
    return image_a;
}

Mat_<uchar> transformLUX(Mat_<Vec3b> image_BGR)
{
    Mat_<uchar> U(image_BGR.size());
    
    int B = 0, G = 0, R = 0, L_int = 0, u_int = 0;
    double L = 0.0, u = 0.0;
    for(int i = 0; i < image_BGR.rows; ++i)
    {
        for(int j = 0; j < image_BGR.cols; ++j)
        {
            B = image_BGR(i, j)[0];
            G = image_BGR(i, j)[1];
            R = image_BGR(i, j)[2];

            L = (pow(R+1, 0.3) * pow(G+1, 0.6) * pow(B+1, 0.1)) - 1;
            L_int = round(L);
            
            if(R > L_int)
            {
                u = (256 * (L_int+1)) / (R + 1);
                u_int = round(u);
                U.at<uchar>(i, j) = u_int;
            }
            else
                U.at<uchar>(i, j) = 255;
        }
    }
    return U;
}

Mat_<uchar> transformModifiedLUX(Mat_<Vec3b> image_BGR)
{
    Mat_<uchar> Ucap(image_BGR.size());
    
    int B = 0, G = 0, R = 0, u_cap_int = 0;
    double u_cap = 0.0;
    for(int i = 0; i < image_BGR.rows; ++i)
    {
        for(int j = 0; j < image_BGR.cols; ++j)
        {
            B = image_BGR(i, j)[0];
            G = image_BGR(i, j)[1];
            R = image_BGR(i, j)[2];

            if(R > G)
            {
                u_cap = (256*G) / R;
                u_cap_int = round(u_cap);
                Ucap.at<uchar>(i, j) = u_cap_int;
            }
            else
                Ucap.at<uchar>(i, j) = 255;
        }
    }
    return Ucap;
}
