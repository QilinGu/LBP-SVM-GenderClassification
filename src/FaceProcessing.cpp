// Revision history:
// [20151211_Curtis] 1. fix several bugs related to cv::Mat
// [20151214_Curtis] 1. add function to return detected facial landmarks

#include "FaceProcessing.h"

CFaceProcessing::CFaceProcessing(const char* faceXml, const char* eyeXml, const char* glassXml, const char* landmarkDat)
   : m_normalFaceSize(128)
{
   if (!m_faceCascade.load(faceXml))
   {
      printf("Error: cannot load xml file for face detection in function CFaceProcessing::CFaceProcessing()\n");
   };

   if (!m_faceFeatureCascade1.load(eyeXml))
   {
      printf("Error: cannot load xml file for eye detection in function CFaceProcessing::CFaceProcessing()\n");
   }

   if (!m_faceFeatureCascade2.load(glassXml))
   {
      printf("Error: cannot load xml file for eye(glass) detection in function CFaceProcessing::CFaceProcessing()\n");
   }

   dlib::deserialize(landmarkDat) >> m_shapePredictor;
}

CFaceProcessing::~CFaceProcessing()
{
}

void CFaceProcessing::FaceHistogramEqualization(cv::Mat& faceImg)
{
   // the following code is copied from book "Mastering OpenCV with Practical Computer Vision Projectss"

   int w = faceImg.cols;
   int h = faceImg.rows;
   cv::Mat wholeFace;
   cv::equalizeHist(faceImg, wholeFace);
   int midX = w / 2;
   cv::Mat leftSide = faceImg(cv::Rect(0, 0, midX, h));
   cv::Mat rightSide = faceImg(cv::Rect(midX, 0, w - midX, h));
   cv::equalizeHist(leftSide, leftSide);
   cv::equalizeHist(rightSide, rightSide);

   for (int y = 0; y<h; y++)
   {
      for (int x = 0; x<w; x++)
      {
         int v;
         if (x < w / 4)
         {
            // Left 25%: just use the left face.
            v = leftSide.at<uchar>(y, x);
         }
         else if (x < w * 2 / 4)
         {
            // Mid-left 25%: blend the left face & whole face.
            int lv = leftSide.at<uchar>(y, x);
            int wv = wholeFace.at<uchar>(y, x);
            // Blend more of the whole face as it moves
            // further right along the face.
            float f = (x - w * 1 / 4) / (float)(w / 4);
            v = (int)((1.0f - f) * lv + (f)* wv + 0.5);
         }
         else if (x < w * 3 / 4)
         {
            // Mid-right 25%: blend right face & whole face.
            int rv = rightSide.at<uchar>(y, x - midX);
            int wv = wholeFace.at<uchar>(y, x);
            // Blend more of the right-side face as it moves
            // further right along the face.
            float f = (x - w * 2 / 4) / (float)(w / 4);
            v = (int)((1.0f - f) * wv + (f)* rv + 0.5);
         }
         else
         {
            // Right 25%: just use the right face.
            v = rightSide.at<uchar>(y, x - midX);
         }
         faceImg.at<uchar>(y, x) = v;
      }// end x loop
   }//end y loop
}

int CFaceProcessing::FaceDetection(const cv::Mat colorImg)
{
   //if (!m_faceCascade.load("D:/Software/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml"))
   //{
   //   printf("Error: cannot load xml file for face detection in function CFaceProcessing::CFaceProcessing()\n");
   //   return 0;
   //};
   
   // color space conversion
   cv::Mat yCbCrImg;
   cv::cvtColor(colorImg, yCbCrImg, CV_RGB2YCrCb);

   // copy first channel of image in YCbCr to gray image 
   m_grayImg = cv::Mat(yCbCrImg.size(), CV_8UC1);
   int fromTo[] = { 0, 0 };
   cv::mixChannels(&yCbCrImg, 1, &m_grayImg, 1, &fromTo[0], 1);

   // segmentation for skin color
   cv::Mat skinBinImg;
   cv::inRange(yCbCrImg, cv::Scalar(0, 85, 135), cv::Scalar(255, 135, 180), skinBinImg);

   // erode and dilate to remove small segmentation
   cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
   cv::erode(skinBinImg, skinBinImg, kernel, cv::Point(-1, -1), 2);
   cv::dilate(skinBinImg, skinBinImg, kernel, cv::Point(-1, -1), 2);
   // apply GaussianBlur to have a complete segmentation
   cv::GaussianBlur(skinBinImg, skinBinImg, cv::Size(0, 0), 3.0);

   // -----------------------------------------------
   // face detection with OpenCV on skin-color region
   // -----------------------------------------------
   cv::Mat skinSegImg;
   cv::Mat skinSegGrayImg(skinSegImg.size(), CV_8UC1);
   m_grayImg.copyTo(skinSegGrayImg, skinBinImg);        
   m_faceCascade.detectMultiScale(skinSegGrayImg, m_faces, 1.2, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(20, 20));

   // eye detection
   EyeDetection();
   
   return m_faces.size();
}

std::vector<cv::Rect>& CFaceProcessing::GetFaces()
{
   return m_faces;
}

int CFaceProcessing::EyeDetection()
{
   // before calling this function, make sure function "FaceDetection" has been called
   
   m_faceStatus.resize(m_faces.size(), 0);

   for (unsigned int i = 0; i < m_faces.size() || i < 0; i++)
   {
      cv::Mat faceImg;
      m_grayImg(m_faces[i]).copyTo(faceImg);

      // histogram equalization on face
      cv::equalizeHist(faceImg, faceImg);

      std::vector<cv::Rect> faceFeature;
      m_faceFeatureCascade1.detectMultiScale(faceImg, faceFeature, 1.2, 3, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(4, 4));

      if (faceFeature.size() != 0)
      {
         m_faceFeatureCascade2.detectMultiScale(faceImg, faceFeature, 1.2, 3, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(4, 4));
         if (faceFeature.size() != 0) m_faceStatus[i] = 1;
         else m_faceStatus[i] = 0;
      }
      else
      {
         m_faceStatus[i] = 0;
      }
   }

   return m_faces.size();
}

int CFaceProcessing::AlignFaces2D(std::vector<cv::Mat>& alignedFaces, bool onlyLargest)
{
   // before calling this function, make sure function "FaceDetection" has been called

   std::vector<cv::Rect> faces;
   
   // find the largest face
   if (onlyLargest == true)
   {
      int idx = GetLargestFace();
      if (idx >= 0) faces.push_back(m_faces[idx]);
   }
   else faces = m_faces;  

   // landmark detection on faces
   std::vector<dlib::full_object_detection> shapes;
   shapes.resize(faces.size());
   dlib::cv_image<unsigned char> dlib_img(m_grayImg);
   m_landmarks.resize(faces.size());
   for (int i = 0; i < (int)faces.size(); i++)
   {
      shapes[i] = m_shapePredictor(dlib_img, dlib::rectangle(faces[i].x, faces[i].y, faces[i].x + faces[i].width - 1, faces[i].y + faces[i].height - 1));

      // [20151214_Curtis] retrieve facial landmarks
      int partsNum = shapes[i].num_parts();
      m_landmarks[i].resize(partsNum);
      for (int j = 0; j < partsNum; j++)
      {
         m_landmarks[i][j].x = (shapes[i].part(j)).x();
         m_landmarks[i][j].y = (shapes[i].part(j)).y();
      }
   }

   // normalize the size of faces
   alignedFaces.resize(faces.size());
   dlib::array<dlib::array2d<unsigned char> > faceChips;
   dlib::extract_image_chips(dlib_img, dlib::get_face_chip_details(shapes, m_normalFaceSize), faceChips);
   for (unsigned int i = 0; i < faces.size(); i++)
   {
      dlib::toMat(faceChips[i]).copyTo(alignedFaces[i]);
   }

   return alignedFaces.size();
}

int CFaceProcessing::GetLargestFace()
{
   int largestIdx = -1;
   int largestArea = 0;
   for (unsigned int i = 0; i < m_faces.size(); i++)
   {
      if (!m_faceStatus[i]) continue;
      int area = m_faces[i].width * m_faces[i].height;
      if (largestArea < area)
      {
         largestIdx = i;
         largestArea = area;
      }
   }

   return largestIdx;
}

std::vector<cv::Point>& CFaceProcessing::GetLandmarks(const unsigned int idx)
{
   // must make sure the idx is valid by yourself before calling this function
   
   return m_landmarks[idx];
}

cv::Mat& CFaceProcessing::GetGrayImages()
{
   return m_grayImg;
}

int CFaceProcessing::FindLandmarksWhichFaces(const std::vector<cv::Point2f>::iterator& landmark, const int n)
{
   int faceIdx = -1;

   for (unsigned int i = 0; i < m_faces.size(); i++)
   {
      int vote = 0;
      for (int j = 0; j < n; j++)
      {
         cv::Point pt = *(landmark + j);
         if (m_faces[i].contains(pt) == true) vote++;
      }

      if (vote >= (n >> 1)) // (TBD) 1/2 landmaks must be tracked for now
      {
         faceIdx = (int)i;
         break;
      }
   }
   
   return faceIdx;
}

std::vector<unsigned char> CFaceProcessing::GetFaceStatus()
{
   return m_faceStatus;
}

bool CFaceProcessing::IncFaceStatus(const int idx, const int val)
{
   if (m_faceStatus.size() < idx) return false;

   m_faceStatus[idx] += val;
   if (m_faceStatus[idx] > 200) m_faceStatus[idx] = 200;
   
   return true;
}
