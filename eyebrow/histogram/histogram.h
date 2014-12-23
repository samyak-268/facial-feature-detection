#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H

#include "opencv2/core/core.hpp"
using namespace std;
using namespace cv;

class Histogram
{
    private:
        Mat intensity_plane;
        vector<int> frequency_histogram;
        vector<double> histogram;
        vector<uchar> transformation_map;
        Mat equalized_image;
        vector<double> equalized_histogram;

    public:
        Histogram(const Mat& _intensity_plane);
        Histogram(const Histogram& _obj);
};

#endif
