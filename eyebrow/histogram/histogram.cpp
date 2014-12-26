#ifndef _HISTOGRAM_CPP
#define _HISTOGRAM_CPP

#include "histogram.h"
#include "opencv2/core/core.hpp"

#define HISTOGRAM_SIZE 256

using namespace std;
using namespace cv;

Histogram::Histogram(const Mat& _intensity_plane)
    :intensity_plane(_intensity_plane)
{
    frequency_histogram.resize(HISTOGRAM_SIZE, 0);
    probability_histogram.resize(HISTOGRAM_SIZE, 0.0);
    transformation_map.resize(HISTOGRAM_SIZE, 0);
    equalized_intensity_plane.create(_intensity_plane.size(), _intensity_plane.type());
    equalized_histogram.resize(HISTOGRAM_SIZE, 0.0);
}

Histogram::Histogram(const Histogram& _obj)
{
    intensity_plane = _obj.intensity_plane;
    frequency_histogram = _obj.frequency_histogram;
    probability_histogram = _obj.probability_histogram;
    transformation_map = _obj.transformation_map;
    equalized_intensity_plane = _obj.equalized_intensity_plane;
    equalized_histogram = _obj.equalized_histogram;
}

void Histogram::calculateFrequencyHistogram()
{
   for(int i = 0; i < intensity_plane.rows; ++i)
   {
       const uchar* intensity_plane_row = intensity_plane.ptr<uchar>(i);
       for(int j = 0; j < intensity_plane.cols; ++j)
           frequency_histogram[intensity_plane_row[j]] += 1;
   }
}

void Histogram::calculateHistogram()
{
    int total_pixels = (intensity_plane.rows * intensity_plane.cols);
    
    calculateFrequencyHistogram();
    for(int i = 0; i < HISTOGRAM_SIZE; ++i)
        probability_histogram[i] = (double)frequency_histogram[i]/total_pixels;
}

void Histogram::calculateTransformationMap()
{
    calculateHistogram();
    
    double cumulative_frequency_sum = 0.0;
    for(int i = 0; i < HISTOGRAM_SIZE; ++i)
    {
        cumulative_frequency_sum += probability_histogram[i];
        transformation_map[i] = round(cumulative_frequency_sum * (HISTOGRAM_SIZE-1));
    }
}

Mat Histogram::constructEqualizedImage()
{
    for (int i = 0; i < intensity_plane.rows; ++i)
    {
        const uchar* intensity_plane_row = intensity_plane.ptr<uchar>(i);
        uchar* equalized_intensity_plane_row = equalized_intensity_plane.ptr<uchar>(i);
        for(int j = 0; j < intensity_plane.cols; ++j)
            equalized_intensity_plane_row[j] = transformation_map[intensity_plane_row[j]];
    }
    return equalized_intensity_plane;
}

vector<double> Histogram::calculateEqualizedHistogram()
{
    return equalized_histogram;
}

vector<int> Histogram::frequencyHistogram()
{
    return frequency_histogram;
}

vector<double> Histogram::probabilityHistogram()
{
    return probability_histogram;
}

vector<uchar> Histogram::transformationMap()
{
    return transformation_map;
}

Mat Histogram::equalizedImage()
{
    return equalized_intensity_plane;
}

vector<double> Histogram::equalizedHistogram()
{
    return equalized_histogram;
}

#endif

