cmake_minimum_required(VERSION 2.8)
project(MouthDetect)

find_package(OpenCV REQUIRED)
add_executable(MouthDetect mouth.cpp)
target_link_libraries(MouthDetect ${OpenCV_LIBS})
