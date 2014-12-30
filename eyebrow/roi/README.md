# The roi module

## Documentation



## Example Usage
```
#include "eyebrow_roi.h"

/* Take the following as input from the user: 
 * string input_image_path
 * string face_cascade_path
 * string eye_cascade_path 
 */

Mat image_BGR = imread(input_image_path);

EyebrowROI eyebrow_detector(image_BGR, face_cascade_path, eye_cascade_path);
eyebrow_detector.detectEyebrows();
eyebrow_detector.displayROI();

```
