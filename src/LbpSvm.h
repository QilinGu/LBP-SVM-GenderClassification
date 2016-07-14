// [20151224_Curtis] include "MultiscaleLbp.h" instead of "lbpfeatures.hpp" 

#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
//#include "lbpfeatures.hpp"
#include <dlib/opencv/cv_image.h>
#include "MultiscaleLbp.h"

class CLbpSvm
{
private:
   //LBPFeatures m_lbpf;
   CvSVM m_svm;
   CvBoost m_boost;
   int m_blockNum;
   int m_scaleMax;
   std::vector<float> m_min;
   std::vector<float> m_max;
   std::vector<int> selectedFeature;
   void FeatureSelect(float* genderLbp, float* genderLabel, std::vector<int>& featureVote, int scaleNum);
   bool ComputeLbp(float* genderLbp, const std::vector<cv::Mat>& maleImgs, const std::vector<cv::Mat>& femaleImgs, int scaleNum);
   int weakCount;
   int trainingDataNum;
   bool surrogate;
public:
   CLbpSvm();
   CLbpSvm(const char* svmModelStr, const char* minMaxStr);
   ~CLbpSvm();
   bool Train(const char* maleImgListStr, const char* femaleImgListStr, const char* svmModelStr);
   bool STrain(const char* maleImgListStr, const char* femaleImgListStr, const char* svmModelStr);
   float Predict(const cv::Mat& croppedGrayFaceImg);
};

