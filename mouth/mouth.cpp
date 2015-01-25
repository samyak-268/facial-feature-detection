/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 *
 */

#include <iostream>
#include <cfloat>
#include <climits>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Mat_<uchar> transformPseudoHue(Mat_<Vec3b> image);
pair<double, double> Stats(Mat_<double> pseudo_hue_plane);

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "Paramters missing\n";
        return -1;
    }
    
    const string input_image_path = argv[1];
    Mat_<Vec3b> image_BGR = imread(input_image_path);
    
    /* CIELAB transformation and using the A-channel

    Mat_<Vec3b> image_Lab;
    vector<Mat> channels;
    cvtColor(image_BGR, image_Lab, CV_BGR2Lab);
    split(image_Lab, channels);

    Mat_<uchar> image_a = channels[1];

    imshow("Original-Image", image_BGR);
    imshow("A-Channel", image_a);
    
    */
    
    /* Convert image to YCrCb, equalize the Y-plane and merge back.
     * Then apply the CIELAB transform to the equalized image and use the 
     * A-channel for lip segmentation.
     */
    
    vector<Mat> channels;
    Mat image_eq;

    cvtColor(image_BGR, image_eq, CV_BGR2YCrCb);
    split(image_eq, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, image_eq);
    cvtColor(image_eq, image_eq, CV_YCrCb2BGR);


    Mat_<Vec3b> image_Lab;
    vector<Mat> channels_1;
    cvtColor(image_eq, image_Lab, CV_BGR2Lab);
    split(image_Lab, channels_1);

    Mat_<uchar> image_a = channels_1[1];

    imshow("Original Image", image_BGR);
    imshow("Equalized Image", image_eq);
    imshow("A-channel", image_a);
    // imshow("Pseudo-Hue", pseudo_hue_plane);

    /*
     * Use the pseudo-hue plane transformation to segment lip region.
     */
    Mat_<uchar> pseudo_hue_plane = transformPseudoHue(image_BGR);

    waitKey(0);
    return 0;
}

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
