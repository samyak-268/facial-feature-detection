#ifndef _BGR2HSI_H
#define _BGR2HSI_H

#include "opencv2/core/core.hpp"
using namespace std;
using namespace cv;

class BGR2HSI
{
    private:
        Mat bgr;
        Mat_<Vec3d> true_hsi;
        int Min(int x, int y, int z);

    public:
        BGR2HSI(const Mat& _bgr);
        BGR2HSI(const BGR2HSI& _obj);
        Mat convert();
        Mat_<Vec3d> returnTrueHSI();
};

#endif
