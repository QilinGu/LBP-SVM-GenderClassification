#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>

class CFaceProcessing
{
private:
   cv::CascadeClassifier m_faceCascade;
   cv::CascadeClassifier m_faceFeatureCascade1;
   cv::CascadeClassifier m_faceFeatureCascade2;
   std::vector<cv::Rect> m_faces;
   std::vector<unsigned char> m_faceStatus; // 0 if this face has no enough facial features. Otherwise, this variable indicates how many previous frames the face can be tracked
   cv::Mat m_grayImg;
   dlib::shape_predictor m_shapePredictor;
   unsigned int m_normalFaceSize;
   std::vector<std::vector<cv::Point> > m_landmarks;
   int EyeDetection();
public:  
   CFaceProcessing(const char* faceXml, const char* eyeXml, const char* glassXml, const char* landmarkDat);
   ~CFaceProcessing();
   int FaceDetection(const cv::Mat colorImg);
   std::vector<cv::Rect>& GetFaces();   
   int AlignFaces2D(std::vector<cv::Mat>& alignedFaces, bool onlyLargest = false);
   int GetLargestFace();
   void FaceHistogramEqualization(cv::Mat& faceImg);
   std::vector<cv::Point>& GetLandmarks(const unsigned int idx);
   cv::Mat& GetGrayImages();
   int FindLandmarksWhichFaces(const std::vector<cv::Point2f>::iterator& landmark, const int n);
   std::vector<unsigned char> GetFaceStatus();
   bool IncFaceStatus(const int idx, const int val);
};
