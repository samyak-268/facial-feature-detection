# The bgr2hsi module

## Documentation

This module contains methods to convert a RGB image to the HSI color space and extract the intensity plane. The public methods available to the programmer are described below:

1. ``` BGR2HSI(const Mat& _bgr) ```: The constructor function which takes an RGB image (``` const Mat& _bgr ```) as input. The conversion is performed on this 3-channel input image.

2. ```Mat convert() ```: Performs the conversion of the RGB image into HSI format. The ``` Mat ``` object returned by the method is an 8-bit, 3-channel image (same as the input image ``` const Mat& _bgr ```). The conversion is governed by the following equations:
    + ``` I = (B + G + R)/3 ```
    + ``` S = (I == 0) ? 0 : (I - Min(R, G, B))/I ```
    + ``` H = (B <= G) ? acos(numerator/denominator) : 2*PI - acos(numerator/denominator) ``` where ``` numerator = (2*R - G - B)/2 ``` and ``` denominator = sqrt((R-G)*(R-G) + (R-B)*(G-B)) ```.

3. ``` Mat_<Vec3d> returnTrueHSI()```: The ``` Mat ``` object returned by the ``` convert() ``` method is an 8-bit (3-channel) image. Conversion from a ``` double ``` value (the results obtaned from the conversion equations) to ``` unsigned char ``` (the values stored in the ``` Mat ``` object denoting the result image) leads to loss of information. This method returns the actual result of the conversion in the form of a 3-channel image where the values are represented by a ``` double ```.

4. ``` Mat extractIntensityPlane(const Mat_<Vec3b>& hsi) ```: Returns the plane corresponding to the intensity values (``` I ``` of the ``` HSI ``` image) of the HSI image.


## Example Usage
```
#include "bgr2hsi.h"

Mat image_BGR = imread(input_image_path);

BGR2HSI converter(image_BGR);
Mat image_HSI = converter.convert();
Mat intensity_plane = converter.extractIntensityPlane();

imshow("Intensity-Plane", intensity_plane);

```
