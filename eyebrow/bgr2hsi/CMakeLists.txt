find_package(OpenCV REQUIRED)

add_library(BGR2HSI bgr2hsi.cpp)
target_link_libraries(BGR2HSI ${OPENCV_LIBS})
