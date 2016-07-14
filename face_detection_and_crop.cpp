// Revision history:
// [20151119_Curtis] 1. normalize and crop face images for gender classfication
//                   2. use uniform LBP and SVM to train gender classfier
// [20151120_Curtis] show menu for user to select one of the following three functions: a). face detection and cropping, b). training, and c). prediction 
// [20151123_Curtis] add the 4th function to do gender classification for video input
// [20151209_Curtis] remove the 3rd function which predicts gender from a list of files
// [20151216_Curtis] add facial landmark tracking to improve the stability of face detection

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>
#include "LbpSvm.h"
#include "FaceProcessing.h"
//#include "OpenNiWrapper.h"

#define USE_HISTO_EQUAL

void FeatureSelectionUseBoost()
{
   // ---------------------------------
   // simply test for feature selection
   // ---------------------------------
   // the ground truth is x^2 + y^2 - 25 >= 0
   // sample space is a 20-by-20 rectangle
   const int trainingDataNum = 47980; // the more accurate classifier will be found with more training data 
   cv::Mat trainingData(trainingDataNum, 3, CV_32F); // 32-bit floating
   cv::Mat response(trainingDataNum, 1, CV_32S); // signed integer
   srand(time(NULL));
   for (int j = 0; j < trainingData.rows; j++)
   {
      const float x = (rand() % 201 - 100.0F) / 10.0F;
      const float y = (rand() % 201 - 100.0F) / 10.0F;
      trainingData.at<float>(j, 0) = x;
      trainingData.at<float>(j, 1) = y;
      // noise
      const float z = (rand() % 21 - 10.0F) / 10.F;
      trainingData.at<float>(j, 2) = z;
      // reponse
      response.at<int>(j, 0) = ((x*x + y*y - 25 >= 0) ? 1 : -1);
   }
   // training
   cv::Boost model;
   printf("Start training boost\n");
   model.train(trainingData, CV_ROW_SAMPLE, response, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), CvBoostParams(CvBoost::GENTLE, 100, 0.95, 5, false, 0));
   // show result
   cv::Mat result(200, 200, CV_8UC3);
   for (int j = 0; j < result.rows; j++)
   {
      for (int i = 0; i < result.cols; i++)
      {
         cv::Mat testData(1, 3, CV_32F);
         const float x = (i - result.cols / 2.0F) / 10.0F;
         const float y = (j - result.rows / 2.0F) / 10.0F;
         testData.at<float>(0, 0) = x;
         testData.at<float>(0, 1) = y;
         testData.at<float>(0, 2) = 0.0F;
         float label = model.predict(testData);
         if (label == 1)
         {
            result.at<cv::Vec3b>(j, i)[0] = 255;
            result.at<cv::Vec3b>(j, i)[1] = 0;
            result.at<cv::Vec3b>(j, i)[2] = 0;
         }
         else
         {
            result.at<cv::Vec3b>(j, i)[0] = 0;
            result.at<cv::Vec3b>(j, i)[1] = 0;
            result.at<cv::Vec3b>(j, i)[2] = 255;
         }
      }
   }
   cv::circle(result, cv::Point(100, 100), 50, CV_RGB(0, 255, 0));
   cv::imshow("result", result);
   // compute which features are more important
   printf("Find weak predictor\n");
   printf("Number of trees: %d\n", model.get_weak_predictors()->total);
   CvSeq* weak = model.get_weak_predictors();
   CvSeqReader reader;
   cvStartReadSeq(weak, &reader);
   cvSetSeqReaderPos(&reader, 0);
   std::vector<int> featureVote(3);
   for (int i = 0; i < weak->total; i++)
   {
	   printf("aaaaa\n");
      CvBoostTree* wtree;
      CV_READ_SEQ_ELEM(wtree, reader);
	  cv::Mat VarImportance = wtree->get_var_importance();
	  std::cout << VarImportance.row(0) << std::endl;
	  int cols = VarImportance.cols;
	  double max_importance = 0;
	  int importance_idx;
	  for (int j = 0; j < cols; j++){
		  if (VarImportance.at<double>(0, j) * 10000 > max_importance){
			  max_importance = VarImportance.at<double>(0, j) * 10000;
			  importance_idx = j;
		  }
	  }
	  //std::cout << VarImportance.row(0) <<" type "<<VarImportance.type()<< std::endl;
	  //printf("fcukme %f ,%f ,%f \n", VarImportance.at<double>(0, 0) * 1000, VarImportance.at<double>(0, 1) * 1000, VarImportance.at<double>(0, 2) * 1000);
      const CvDTreeNode* node = wtree->get_root();
      CvDTreeSplit* split = node->split;
	  //int index = split->condensed_idx;
	  int index = split->var_idx;
	  featureVote[index]++;
	  printf("%d, %d, %d\n", split->condensed_idx, split->var_idx, importance_idx);
   }
   printf("Vote: %d, %d, %d\n", featureVote[0], featureVote[1], featureVote[2]);
   // remove the weak classifiers which use the least important fature (we always remove the last feature in this simulation)
   weak = model.get_weak_predictors();
   cvStartReadSeq(weak, &reader);
   cvSetSeqReaderPos(&reader, 0);
   for (int i = 0; i < weak->total; i++)
   {
      CvBoostTree* wtree;
      CV_READ_SEQ_ELEM(wtree, reader);
      const CvDTreeNode* node = wtree->get_root();
      CvDTreeSplit* split = node->split;
      const int index = split->condensed_idx;

      // 1). remove all weak classifiers except the first one
      //if (i > 0)
      //{
      //   model.prune(cvSlice(i, i + 1));
      //   i--;
      //}
      // 2). remove the weakest classifiers
      if (index == 2)
      {
         model.prune(cvSlice(i, i + 1));
         i--;
      }
   }
   // show result after pruning
   cv::Mat resultPrune(200, 200, CV_8UC3);
   for (int j = 0; j < resultPrune.rows; j++)
   {
      for (int i = 0; i < resultPrune.cols; i++)
      {
         cv::Mat testData(1, 3, CV_32F);
         const float x = (i - resultPrune.cols / 2.0F) / 10.0F;
         const float y = (j - resultPrune.rows / 2.0F) / 10.0F;
         testData.at<float>(0, 0) = x;
         testData.at<float>(0, 1) = y;
         testData.at<float>(0, 2) = 0.0F;
         float label = model.predict(testData);
         if (label == 1)
         {
            resultPrune.at<cv::Vec3b>(j, i)[0] = 255;
            resultPrune.at<cv::Vec3b>(j, i)[1] = 0;
            resultPrune.at<cv::Vec3b>(j, i)[2] = 0;
         }
         else
         {
            resultPrune.at<cv::Vec3b>(j, i)[0] = 0;
            resultPrune.at<cv::Vec3b>(j, i)[1] = 0;
            resultPrune.at<cv::Vec3b>(j, i)[2] = 255;
         }
      }
   }
   cv::circle(resultPrune, cv::Point(100, 100), 50, CV_RGB(0, 255, 0));
   cv::imshow("resultPrune", resultPrune);
   // show
   cv::waitKey();
   return;
}

int main(int argc, const char** argv)
{  
   //FeatureSelectionUseBoost();
   //return 0;
   
   // show menu
   printf("--------------------------------------\n");
   printf("Select one of the following functions:\n");
   printf("[1] face detection and cropping\n");
   printf("[2] gender classifier training\n");
   printf("[3] gender classification (video)\n");
   printf("--------------------------------------\n");
   char key = getchar();
   // ---------------------------
   // face detection and cropping
   // ---------------------------
   if (key == '1')
   {
      // modify this for different database
      std::ifstream fileIndex("D:/face_detection/positive_database/lfw/faceImgList.csv");

      // load the cascades and landmark model
      CFaceProcessing fp("D:/Software/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml",
      "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml",
      "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml",
      "D:/Vision_Project/shape_predictor_68_face_landmarks.dat");

      // load trained SVM
      CLbpSvm lbpSvm("./svm.model", "./MinMax.csv");
      
      // read next image until no image can be read
      while (!fileIndex.eof())
      {
         char faceImgName[256] = { 0 };
         fileIndex.getline(faceImgName, 256);
         printf("%s\n", faceImgName);
         cv::Mat img = cv::imread(faceImgName);

         // face detection
         std::vector<cv::Rect> faces;
         int faceNum= fp.FaceDetection(img);
         cv::Mat croppedImg;
         if (faceNum > 0)
         {
            faces = fp.GetFaces();

            // normalize the largest face image with landmark
            int facelargestIdx = fp.GetLargestFace();
            if (facelargestIdx >= 0)
            {
               std::vector<cv::Mat> largestNormalizedImg;
               fp.AlignFaces2D(largestNormalizedImg, true);

               // crop and show the cropped face
               float gender = 0.0;
               if (largestNormalizedImg.empty() == false)
               {
                  largestNormalizedImg[0].copyTo(croppedImg);
#ifdef USE_HISTO_EQUAL
                  cv::equalizeHist(croppedImg, croppedImg);
#endif
                  cv::imshow("largest cropped image", croppedImg);
                  gender = lbpSvm.Predict(croppedImg);
               }

               // show detected faces and the original image 
               for (unsigned int i = 0; i < faces.size(); i++)
               {
                  cv::rectangle(img, faces[i], CV_RGB(180, 180, 180), 2);
               }

               // show gender of the largest face
               //if (gender > 0.0) cv::rectangle(img, faces[facelargestIdx], CV_RGB(0, 0, 255), 2);
               //else if (gender < 0.0) cv::rectangle(img, faces[facelargestIdx], CV_RGB(255, 0, 0), 2);
               float absResult = abs(gender);
               if (abs(gender) >= 0.002) // 0.001 ~ 0.002
               {
                  if (gender <= 0)
                  {
                     cv::rectangle(img, faces[facelargestIdx], CV_RGB(0, 0, 255), 2); // male
                  }
                  else
                  {
                     cv::rectangle(img, faces[facelargestIdx], CV_RGB(255, 0, 0), 2); // female
                  }
               }
               // debug - show faces with low confidence of gender classification 
               else
               {
                  cv::rectangle(img, faces[facelargestIdx], CV_RGB(200, 200, 200), 2);
               }
               //
               printf("Gender confidence = %f\n", gender);
            }
            cv::imshow(faceImgName, img);

            // key processing
            int key = cv::waitKey(0);
            if (key == 27) break;
            else if (key == 102) // 'f' for female
            {
               char faceGenderImgName[256] = { 0 };
               memcpy(faceGenderImgName, faceImgName, 256 * sizeof(char));
               int strLen = strlen(faceGenderImgName);
               memcpy(&faceGenderImgName[strLen - 4], "_female.bmp", 11 * sizeof(char));
               if (croppedImg.empty() == false) cv::imwrite(faceGenderImgName, croppedImg);
            }
            else if (key == 109) // 'm' for male
            {
               char faceGenderImgName[256] = { 0 };
               memcpy(faceGenderImgName, faceImgName, 256 * sizeof(char));
               int strLen = strlen(faceGenderImgName);
               memcpy(&faceGenderImgName[strLen - 4], "_male.bmp", 9 * sizeof(char));
               if (croppedImg.empty() == false) cv::imwrite(faceGenderImgName, croppedImg);
            }

            cv::destroyAllWindows();
         }
      }
   }
   
   // --------------------------
   // gender classifier training
   // --------------------------
   else if (key == '2') 
   { 
      CLbpSvm lbpSvm;
      //lbpSvm.Train("D:/face_detection/positive_database/lfw_cropped_face/maleImgList.csv", "D:/face_detection/positive_database/lfw_cropped_face/femaleImgList.csv", "./svm.model");
	  //[20160104_Sylar]
	  bool flag = lbpSvm.STrain("C:/Users/TB560074/Desktop/lfw_cropped_face/maleImgList.csv", "C:/Users/TB560074/Desktop/lfw_cropped_face/femaleImgList.csv", "./SvmBoostMScale.model");
	  if (!flag)
		  printf("Train failed\n");
   }

   // -----------------------------
   // gender classification (video)
   // -----------------------------
   else if (key == '3')
   {
      // load SVM model
	  //[20160104_Sylar]
	  //CLbpSvm lbpSvm("./Test/svm.model", "./Test/MinMax.csv");
	  CLbpSvm lbpSvm("./Test/SvmBoostMScale.model", "./Test/MinMax.csv");
      // use camera, ASUS xtion, or video as input source
      //cv::VideoCapture cap(0);
      //COpenNiWrapper openNiWrapper;
      cv::VideoCapture cap("C:/Users/TB560074/Desktop/face_detection_and_crop/Backstreet.mp4");
      if (!cap.isOpened()) return -1;
      
      // load the cascades
      /*CFaceProcessing fp("D:/Software/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml",
      "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml",
      "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml",
      "D:/Vision_Project/shape_predictor_68_face_landmarks.dat");*/

	  //[20160104_Sylar]
	  CFaceProcessing fp("C:/Program Files/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml",
		  "C:/Program Files/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml",
		  "C:/Program Files/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml",
		  "C:/Users/TB560074/Desktop/face_detection_and_crop/face_detection_and_crop/shape_predictor_68_face_landmarks.dat");
  
      // main loop
      cv::Mat img;
      bool showLandmark = false;
      bool showCroppedFaceImg = false;
      cv::Mat grayFrame;
      cv::Mat grayFramePrev;
      std::vector<std::vector<cv::Point>> fLandmarksPrev;
      std::vector<std::vector<cv::Point>> fLandmarks;
      std::vector<unsigned char> faceStatusPrev;
      std::vector<float> accGenderConfidencePrev;
	  float totalCount = 0;
	  float falseCount = 0;
	  std::vector<cv::Mat> prevCropped;
      while (1)
      {  
         //openNiWrapper.GetDepthColorRaw();
         //openNiWrapper.ConvertDepthColorRawToImage(cv::Mat(), img);
         cap >> img;
         if (img.empty()) break;

         // (optional) backup original image for offline debug
         cv::Mat originImg(img.size(), img.type());
         img.copyTo(originImg);

         // time calculation
         unsigned long sTime = clock();
         unsigned long eTime = clock();
        
         // --------------
         // face detection 
         // --------------
         std::vector<cv::Rect> faces;
         int faceNum = fp.FaceDetection(img);
         std::vector<cv::Mat> croppedImgs;
         if (faceNum > 0)
         {
            faces = fp.GetFaces();

            // normalize the face image with landmark
            std::vector<cv::Mat> normalizedImg;
            fp.AlignFaces2D(normalizedImg);

            // ----------------------------------------
            // crop faces and do histogram equalization
            // ----------------------------------------
            croppedImgs.resize(faceNum);
            for (int i = 0; i < faceNum; i++)
            {
				cv::Mat temp;
				normalizedImg[i].copyTo(croppedImgs[i]);
#ifdef USE_HISTO_EQUAL
				//cv::equalizeHist(croppedImgs[i], croppedImgs[i]);
#endif
            }

            // ---------------------------------
            // extraction landmarks on each face
            // ---------------------------------
            fLandmarks.resize(faceNum);
            for (int i = 0; i < faceNum; i++)
            {
               fLandmarks[i] = fp.GetLandmarks(i);
            }
         }
         // (debug) show no face 
         //else
         //{
         //   printf("Detect no face\n\n");
         //}

         // ----------------------
         // track facial landmarks
         // ----------------------
         grayFrame = fp.GetGrayImages();
         std::vector<std::pair<int, int>> trackFromTo;
         if (grayFramePrev.empty() == false && fLandmarksPrev.size() != 0) // do tracking when the current frame is not the first one
         {
            std::vector<cv::Point2f> ptsPrev;
            std::vector<cv::Point2f> pts;
            // 2d vector to 1d vector
            for (unsigned int i = 0; i < fLandmarksPrev.size(); i++)
            {
               ptsPrev.insert(ptsPrev.end(), fLandmarksPrev[i].begin(), fLandmarksPrev[i].end());
            }
            std::vector<unsigned char> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(grayFramePrev, grayFrame, ptsPrev, pts, status, err);
            
            // (debug) show tracked facial landmarks
            //for (unsigned int i = 0; i < pts.size(); i++)
            //{
            //   cv::circle(img, pts[i], 1, CV_RGB(255, 255, 255));
            //}
            
            // check if the tracked facial landmarks are located in a certain face
            int offset = 0;
            for (unsigned int i = 0; i < fLandmarksPrev.size(); i++)
            {
               // previous frame --> current frame
               //       i        -->    faceIdx
               int faceIdx = fp.FindLandmarksWhichFaces(pts.begin() + offset, fLandmarksPrev[i].size());
               if (faceIdx != -1)
               {
                  fp.IncFaceStatus(faceIdx, (int)faceStatusPrev[i]);
                  trackFromTo.push_back(std::pair<int, int>(i, faceIdx));
               }
               offset += fLandmarksPrev[i].size();
            }
         }
         
         // ----------------------------
         // (debug) show faces and count 
         // ----------------------------
         //if (faceNum > 0)
         //{
         //   faces = fp.GetFaces();
         //   std::vector<unsigned char> status = fp.GetFaceStatus();
         //   for (int i = 0; i < faceNum; i++)
         //   {
         //      if (status[i])
         //      {
         //         cv::rectangle(img, faces[i], CV_RGB(200, 200, 200), 2); // face with eyes
         //         cv::putText(img, std::to_string((int)status[i]), cv::Point(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2), 1, cv::FONT_HERSHEY_COMPLEX, CV_RGB(0, 0, 0), 2);
         //      }
         //      else
         //      {
         //         cv::rectangle(img, faces[i], CV_RGB(50, 50, 50), 2); // face with eyes
         //         cv::putText(img, std::to_string((int)status[i]), cv::Point(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2), 1, cv::FONT_HERSHEY_COMPLEX, CV_RGB(0, 0, 0), 2);
         //      }
         //   }
         //}
         
         // --------------------------------------------
         // do gender classification and display results
         // --------------------------------------------
         std::vector<unsigned char> status = fp.GetFaceStatus();
         for (int i = 0; i < faceNum; i++)
         {
            if (status[i])
            {
               float result = lbpSvm.Predict(croppedImgs[i]);
               // display face and gender
               float absResult = abs(result);
               
               // show faces when the current confidence is high enough
               if (absResult >= 0.001) // 0.001 ~ 0.002
               {
				   totalCount++;
                  if (result <= 0) // male
                  {
                     char beliefStr[64] = { 0 };
                     sprintf(beliefStr, "%f", absResult);
                     cv::putText(img, beliefStr, cv::Point(faces[i].x, faces[i].y + faces[i].height + 30), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0, 0, 255));
                     cv::rectangle(img, faces[i], CV_RGB(0, 0, 255), 2); // male
                  }
                  else // female
                  {
                     char beliefStr[64] = { 0 };
                     sprintf(beliefStr, "%f", absResult);
                     cv::putText(img, beliefStr, cv::Point(faces[i].x, faces[i].y + faces[i].height + 30), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 0));
                     cv::rectangle(img, faces[i], CV_RGB(255, 0, 0), 2); // female
					 falseCount++;
					 /*cv::imshow("FALSE", croppedImgs[i]);
					 for (int it = 0; it < (int)prevCropped.size(); it++)
						 cv::imshow(std::to_string(it), prevCropped[it]);
					 cv::waitKey();*/
                  }
               }

               // -----------------------------
               // (debug) show facial landmarks
               // -----------------------------
               if (showLandmark == true)
               {
                  for (int i = 0; i < faceNum; i++)
                  {
                     if (status[i])
                     {
                        for (unsigned int j = 0; j < fLandmarks[i].size(); j++)
                        {
                           cv::circle(img, fLandmarks[i][j], 1, CV_RGB(255, 255, 255), 1);
                        }
                     }
                  }
               }

               // --------------------------------
               // (debug) show cropped face images
               // --------------------------------
               if (showCroppedFaceImg == true)
               {
                  for (int i = 0; i < faceNum; i++)
                  {
                     if (status[i])
                     {
                        cv::imshow(std::to_string(i), croppedImgs[i]);
                     }
                  }
               }
            }
         }
		 //-------------------------------------
		 // For false classification's prev face
		 //-------------------------------------
		 prevCropped.clear();
		 prevCropped.resize(faceNum);
		 for (int i = 0; i < faceNum; i++){
			 croppedImgs[i].copyTo(prevCropped[i]);
		 }
         // ----------------------------------------------------
         // current data will be previous data in the next frame
         // ----------------------------------------------------
         if (faceNum > 0)
         {
            grayFrame.copyTo(grayFramePrev);
            fLandmarksPrev.resize(fLandmarks.size());
            for (unsigned int i = 0; i < fLandmarks.size(); i++)
            {
               fLandmarksPrev[i] = fLandmarks[i];
            }
            faceStatusPrev = fp.GetFaceStatus();
         }
         else
         {
            grayFramePrev = cv::Mat();
            fLandmarksPrev.clear();
            faceStatusPrev.clear();
         }
 
         // show processing time
         eTime = clock();
         char deltaTimeStr[256] = { 0 };
         sprintf_s(deltaTimeStr, "%d ms", (eTime - sTime));
         cv::putText(img, deltaTimeStr, cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255));
      
         cv::imshow("Result", img);
         int key = 0;
         //if (faceNum > 0) key = cv::waitKey();
         //else key = cv::waitKey(1);
         key = cv::waitKey(1);

         if (key == 27) break;
         else if (key == 83 || key == 115)
         {
            std::time_t time = std::time(NULL);
            char timeStr[128] = { 0 };
            std::strftime(timeStr, sizeof(timeStr), "./Offline/%Y-%m-%d-%H-%M-%S.bmp", std::localtime(&time));
            cv::imwrite(timeStr, originImg);
         }
         else if (key == 76 || key == 108) // 'l' or 'L'
         {
            showLandmark = !showLandmark;
         }
         else if (key == 70 || key == 102) // 'f' or 'F'
         {
            showCroppedFaceImg = !showCroppedFaceImg;
         }
      }

	  std::cout << "False Rate :"<<falseCount <<"/"<< totalCount << std::endl;
   }

   //
   system("pause");
   return 0;
}
