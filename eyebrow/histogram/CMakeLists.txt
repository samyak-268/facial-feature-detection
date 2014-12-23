find_package(OpenCV REQUIRED)

add_library(HISTOGRAM histogram.cpp)
target_link_libraries(HISTOGRAM ${OPENCV_LIBS})
