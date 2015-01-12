cmake_minimum_required(VERSION 2.8)
project(EyebrowDetect)

include_directories("${PROJECT_SOURCE_DIR}/roi")
add_subdirectory(roi)

find_package(OpenCV REQUIRED)
add_executable(EyebrowDetect eyebrow.cpp)
target_link_libraries(EyebrowDetect ${OpenCV_LIBS})
target_link_libraries(EyebrowDetect EYEBROW_ROI)
