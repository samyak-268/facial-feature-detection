#ifndef _BGR2HSI_H
#define _BGR2HSI_H

#include "opencv2/core/core.hpp"
using namespace std;
using namespace cv;

class BGR2HSI
{
    private:
        Mat bgr;
        Mat_<Vec3d> true_hsi;
        int Min(int x, int y, int z);

    public:
        BGR2HSI(const Mat& _bgr);
        BGR2HSI(const BGR2HSI& _obj);
        Mat convert();
        Mat_<Vec3d> returnTrueHSI();
        Mat extractIntensityPlane(const Mat_<Vec3b>& hsi);
        vector<double> calculateHistogram(const Mat& intensity_plane);
        vector<double> equalizeHistogram(const vector<double>& histogram);

        /* TODO
         *  (1) Modify equalizeHistogram() method to round off the obtained values. The quantity being calculated
         *      is not the equalized histogram, but the transformation function s = T(r) map where 
         *      equalized_histogram[i] (after rounding) is the transformed value s for the input r = i.
         *  
         *  (2) Change the function decalaration to vector<uchar> transformationMap(const vector<double>& histogram).
         *
         *  (3) Add a new method Mat equaliszedImage(const Mat& intensity_plane, const vector<uchar>& transformation_map)
         *      which returns the equalized (enhanced) image after performing the transformation mappings defined by
         *      transformation_map onto the intensity_plane image
         *
         *  (4) Add another similar method to return the equalized histogram from the equalized image
         *
         *  (5) Port all the histogram equalization related  methods and data members to a separate module. 
         *      The data members of the module: 
         *          (a) Mat intensity_plane
         *          (b) vector<double> frequency_histogram
         *          (c) vector<double> histogram
         *          (d) vector<uchar> transformation_map
         *          (e) Mat equalized_image
         *          (f) vector<double> equalized_histogram
         */
};

#endif
