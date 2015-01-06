# The histogram module

## Documentation

This modules contains functions for constructing a histogram from a given intensity plane (``` CV_8UC1 ```) and then performing histogram equalization of the same. The public methods available to the programmer are described below:

1. ``` void calculateFrequencyHistogram() ```: This method calculates the number of pixels in the image corresponding to each intensity level from 0 to 255. The method ``` vector<int> frequencyHistogram()``` returns a vector corresponding to the frequency histogram calculated by this method.

2. ``` void calculateHistogram() ```: This method calculates, for each intensity value between 0 to 255, the fraction of the total pixels that belong to that particular intensity value. In other words, the probability of a pixel to take that intensity value. The method ``` vector<double> probabilityHistogram()``` returns a vector corresponding to the probability (frequency) histogram calculated by this method.

3. ``` void calculateTransformationMap() ```: This method creates the transformation map (representing the transformation function) corresponding to the histogram equalization. For a given input intensity value ``` r ```, the transformation function ``` s = T(r) ``` returns the intensity level ``` s ```in the output image. The method ``` vector<uchar> transformationMap() ``` returns a vector corresponding to the transformation map calculated by this method.

4. ``` Mat constructEqualizedImage() ```: This method utilizes the transformation map created by the ``` void calculateTransformationMap() ``` method to create and return the output, histogram-equalized image. The equalized image can also be retrieved via a call to the ``` Mat equalizedImage() ``` method.

5. ``` vector<double> calculateEqualizedHistogram() ```: This method is used to calculate the final equalized histogram corresponding to the equalized image constructed by ``` Mat constructEqualizedImage() ```. A call to ``` vector<double> equalizedHistogram() ``` returns the equalized histogram computed by this method.


## Example Usage
```
#include "histogram.h"

// The input image (intensity plane) must be 8-bit, single channel
Mat intensity_plane = imread(input_image_path);
Histogram hist_calc_obj(intensity_plane);

// Calcualting and storing the equalized histogram values
hist_calc_obj.calculateEqualizedHistogram();
vector<double> eq_hist = his_calc_obj.equalizedHistogram();

// Constructing and displaying the histogram-equalized image
hist_calc_obj.constructEqualizedImage();
Mat eq_intensity_plane = hist_calc_obj.equalizedImage();
imshow("Equalized-Intensity-Plane", eq_intensity_plane);
```
