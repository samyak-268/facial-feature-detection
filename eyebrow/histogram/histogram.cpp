#ifndef _HISTOGRAM_CPP
#define _HISTOGRAM_CPP

#include "histogram.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

Histogram::Histogram(const Mat& _intensity_plane)
    :intensity_plane(_intensity_plane)
{
    equalized_image.create(_intensity_plane.size(), _intensity_plane.type());
}

Histogram::Histogram(const Histogram& _obj)
{
    intensity_plane = _obj.intensity_plane;
    frequency_histogram = _obj.frequency_histogram;
    histogram = _obj.histogram;
    transformation_map = _obj.transformation_map;
    equalized_image = _obj.equalized_image;
    equalized_histogram = _obj.equalized_histogram;
}

#endif

