find_package(OpenCV REQUIRED)

add_library(EYEBROW_ROI eyebrow_roi.cpp)
target_link_libraries(EYEBROW_ROI ${OPENCV_LIBS})
