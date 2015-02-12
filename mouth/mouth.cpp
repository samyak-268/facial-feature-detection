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
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Mat_<Vec3b> extractFaceROI(Mat_<Vec3b> image, string face_cascade_path);
Mat_<Vec3b> extractMouthROI(Mat_<Vec3b> face_image);

Mat_<Vec3b> equalizeImage(Mat_<Vec3b> image_BGR);
Mat_<uchar> transformPseudoHue(Mat_<Vec3b> image);
pair<double, double> Stats(Mat_<double> pseudo_hue_plane);
Mat_<uchar> transformCIELAB(Mat_<Vec3b> image_BGR);
Mat_<uchar> transformLUX(Mat_<Vec3b> image_BGR);
Mat_<uchar> transformModifiedLUX(Mat_<Vec3b> image_BGR);

pair<double, double> returnImageStats(const Mat_<uchar>& image);
Mat_<uchar> binaryThresholding(const Mat_<uchar>& image, const pair<double, double>& stats);
int returnLargestContourIndex(vector<vector<Point> > contours);

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "Paramters missing\n";
        return -1;
    }
    
    const string input_image_path = argv[1];
    const string face_cascade_path = argv[2];

    Mat_<Vec3b> image_BGR = imread(input_image_path);
    Mat_<Vec3b> mouth = extractMouthROI(extractFaceROI(image_BGR, face_cascade_path));
    
    Mat_<uchar> pseudo_hue_plane = transformPseudoHue(mouth);
    Mat_<uchar> pseudo_hue_bin = binaryThresholding(pseudo_hue_plane, 
            returnImageStats(pseudo_hue_plane));
    
    // A clone image is required because findContours() modifies the input image
    Mat binary_clone = pseudo_hue_bin.clone();
    vector<vector<Point> > contours;
    findContours(binary_clone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    
    // Initialize blank image (for drawing contours)
    Mat_<uchar> image_contour(pseudo_hue_bin.size());
    for(int i = 0; i < image_contour.rows; ++i)
    {
        for(int j = 0; j < image_contour.cols; ++j)
            image_contour.at<uchar>(i, j) = 0;
    }

    // Draw largest contour on the blank image
    cout << "Size of the contour image: " << image_contour.rows << " X " << image_contour.cols << "\n";
    int largest_contour_idx = returnLargestContourIndex(contours);
    for(int i = 0; i < contours[largest_contour_idx].size(); ++i)
    {
        Point_<int> pt = contours[largest_contour_idx][i];
        image_contour.at<uchar>(pt.y, pt.x) = 255;
    }
    
    
    // imshow("Input-Image", image_BGR);
    // imshow("Face-ROI", face);
    // imshow("Mouth-ROI", mouth);
    imshow("Pseudo-Hue", pseudo_hue_plane);
    // imshow("Pseudo-Hue-Binary", pseudo_hue_bin);
    imshow("Contour", image_contour);
    
    /*
     * Mat_<uchar> chrominance_plane = transformLUX(image_BGR);
     * Mat_<uchar> modified_chrominance_plane = transformModifiedLUX(image_BGR);
     *
     * imshow("Chrominance-Plane", chrominance_plane);
     * imshow("Modified-Chrominance-Plane", modified_chrominance_plane);
     */

    waitKey(0);
    return 0;
}

Mat_<Vec3b> extractFaceROI(Mat_<Vec3b> image, string face_cascade_path)
{
    CascadeClassifier face_cascade;
    vector<Rect_<int> > faces;
    
    face_cascade.load(face_cascade_path);
    face_cascade.detectMultiScale(image, faces, 1.15, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

    Mat_<Vec3b> face_ROI;
    for(int i = 0; i < faces.size(); ++i)
    {
        Rect_<int> face = faces[i];
        
        int face_rows = face.height;
        int face_cols = face.width;

        face_ROI = image(Rect(face.x, face.y, face_cols, face_rows));

        /*
        int roi_x = face.x, roi_y = face.y + ((2 * face_rows) / 3);
        int roi_rows = (face_rows - roi_y), roi_cols = face_cols;

        rectangle(image, Point(roi_x, roi_y), Point(roi_x+roi_cols, roi_y+roi_rows),
                                Scalar(255, 0, 0), 1, 4);
        
        */
    }
    return face_ROI;
}

Mat_<Vec3b> extractMouthROI(Mat_<Vec3b> face_image)
{
    int face_rows = face_image.rows;
    int face_cols = face_image.cols;

    int mouth_x = (face_cols / 4), mouth_y = (2 * face_rows) / 3;
    int mouth_rows = (2 * (face_rows - mouth_y)) / 3, mouth_cols = (face_cols / 2);

    return face_image(Rect(mouth_x, mouth_y, mouth_cols, mouth_rows));
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

int returnLargestContourIndex(vector<vector<Point> > contours)
{
    int max_contour_size = 0;
    int max_contour_idx = -1;
    for(int i = 0; i < contours.size(); ++i)
    {
        if(contours[i].size() > max_contour_size)
        {
            max_contour_size = contours[i].size();
            max_contour_idx = i;
        }
    }
    return max_contour_idx;
}
