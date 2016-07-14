// [20151123_Curtis] add a member variable "m_blockNum" for block-based LBP
// [20151224_Curtis] use multi-scale LBP

#include "LbpSvm.h"
#include <queue>
#include <map>

#define FeatureSelection
//==============
const double FACE_ELLIPSE_CY = 0.40; //0.40
const double FACE_ELLIPSE_W = 0.40;  //0.50 0.40      // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.55; //0.8   0.55     // Controls how tall the face mask is.
using namespace cv;
//==============

CLbpSvm::CLbpSvm()
   : m_blockNum(16)
   , m_scaleMax(8)
   , weakCount(500)
   , surrogate(false)
{
   //m_lbpf.initUniform();	
}

CLbpSvm::CLbpSvm(const char* svmModelStr, const char* minMaxStr)
   : m_blockNum(16)
   , m_scaleMax(8)
   , weakCount(500)
   , surrogate(false)
{
   //m_lbpf.initUniform();
   

	//[20160104_Sylar]
	printf("loading...\n");
	m_svm.load(svmModelStr);
	//m_boost.load(svmModelStr);
   m_min.resize(m_blockNum * m_blockNum * 59 * m_scaleMax);
   m_max.resize(m_blockNum * m_blockNum * 59 * m_scaleMax);

   FILE* fptr = fopen(minMaxStr, "r");
   if (fptr == NULL)
   {
      printf("Error: have no min/max file in function CLbpSvm::CLbpSvm()\n");
   }
   else
   {
      for (int i = 0; i < m_blockNum * m_blockNum * 59 * m_scaleMax; i++)
      {
         fscanf(fptr, "%f %f\n", &m_min[i], &m_max[i]);
         // [20151224_Curtis] workaround for unknow reason, min and max are both 0
         if (m_min[i] == 0 && m_max[i] == 0)
         {
            m_min[i] = -1;
            m_max[i] = -1;
         }
         else if (m_min[i] >= m_max[i])
         {
            printf("Error: incorrect min/max file in function CLbpSvm::CLbpSvm()\n");
            break;
         }
      }      
      fclose(fptr);
   }
   FILE* featureptr = fopen("./Test/SelectedFeatures.csv", "r");
   if (!featureptr )
   {
	   printf("Error: have no SelectedFeatures.csv file in function CLbpSvm::CLbpSvm()\n");
   }
   else
   {
	   int n;
	   while (fscanf(featureptr, "%i", &n)!=EOF){		  
		   selectedFeature.push_back(n);
		   //printf("%i\n", n);
	   }
	   std::cout << selectedFeature.size() << std::endl;
	   fclose(featureptr);
   }
}

CLbpSvm::~CLbpSvm()
{
}
//[20160105_Sylar]
bool CLbpSvm::STrain(const char* maleImgListStr, const char* femaleImgListStr, const char* svmModelStr)
{
   // --------------------
   // load all male images
   // --------------------
   std::vector<cv::Mat> maleImgs;
   std::vector<cv::Mat> femaleImgs;
   std::ifstream file(maleImgListStr);
   //=============
   Mat qq, mask, dstImg;
   //=============
   while (!file.eof())
   {
	  /*char faceImgName[256] = { 0 };
      file.getline(faceImgName, 256);
      maleImgs.push_back(cv::imread(faceImgName, 0)); // to grayscale directly*/
	   char faceImgName[256] = { 0 };
	   file.getline(faceImgName, 256);
	   int h = 128;
	   int w = 128;
	   qq = cv::imread(faceImgName, 0);
	   mask = Mat(qq.size(), CV_8U, Scalar(0));
	   Point faceCenter = Point(w / 2, cvRound(h * FACE_ELLIPSE_CY));
	   Size size = Size(cvRound(w * FACE_ELLIPSE_W), cvRound(h * FACE_ELLIPSE_H));
	   ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
	   dstImg = Mat(qq.size(), CV_8U, Scalar(0));
	   qq.copyTo(dstImg, mask);
	   maleImgs.push_back(dstImg);  // to grayscale directly
   }
   // load all female images
   file = std::ifstream(femaleImgListStr);
   while (!file.eof())
   {
      /*char faceImgName[256] = { 0 };
      file.getline(faceImgName, 256);
      femaleImgs.push_back(cv::imread(faceImgName, 0)); // to grayscale directly*/
	   char faceImgName[256] = { 0 };
	   file.getline(faceImgName, 256);
	   int h = 128;
	   int w = 128;
	   qq = cv::imread(faceImgName, 0);
	   mask = Mat(qq.size(), CV_8U, Scalar(0));
	   Point faceCenter = Point(w / 2, cvRound(h * FACE_ELLIPSE_CY));
	   Size size = Size(cvRound(w * FACE_ELLIPSE_W), cvRound(h * FACE_ELLIPSE_H));
	   ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
	   dstImg = Mat(qq.size(), CV_8U, Scalar(0));
	   qq.copyTo(dstImg, mask);
	   femaleImgs.push_back(dstImg);  // to grayscale directly
   }
   //
   printf("Start training from %d male images and %d female images\n", maleImgs.size(), femaleImgs.size());
   //
   if (maleImgs.size() == 0 || femaleImgs.size() == 0)
   {
      printf("Error: must have enough male/female face images in function CLbpSvm::Train()\n");
      return false;
   } 
   // prepare labels for gender
   float* genderLabel = new float[maleImgs.size() + femaleImgs.size()];
   for (unsigned int i = 0; i < maleImgs.size(); i++) // 1 for male
   {
      genderLabel[i] = 1.0;
   }
   for (unsigned int i = 0; i < femaleImgs.size(); i++) // -1 for female
   {
      genderLabel[maleImgs.size() + i] = -1.0;
   }   
   //----------------------------------
   // [20160106_Sylar]
   // MultiScale boost training
   //----------------------------------
   trainingDataNum = maleImgs.size() + femaleImgs.size();
   int scaleNum = 1;
   unsigned int stride = m_blockNum * m_blockNum * 59 ;

   //Save Selected Features
   FILE* SFptr = fopen("SelectedFeatures.csv", "a");
   
   // For svm
   printf("svm Mat initialize\n");
   cv::Mat genderLbpMat;
   cv::Mat genderLabelMat( (int)maleImgs.size() + (int)femaleImgs.size(), 1, CV_32FC1, genderLabel);
   for (; scaleNum <= m_scaleMax; scaleNum++){
	   //Compute single scale Lbp
	   float* genderLbp = new float[(maleImgs.size() + femaleImgs.size()) * m_blockNum * m_blockNum * 59];
	   printf("Compute Lbp Scale %d\n", scaleNum);
	   if (!ComputeLbp(genderLbp, maleImgs, femaleImgs, scaleNum)){
		   return false;
	   }
	   // size of m_blockNum * m_blockNum * 59
	   // To vote for features
	   std::vector<int> featureVote;
	   // record selected feature index
	   std::vector<int> featureIdx;
	   featureIdx.clear();
	   FeatureSelect(genderLbp, genderLabel, featureVote, scaleNum);
	   for (int i = 0; i < featureVote.size(); i++){
		   if (featureVote[i] != 0){
			   fprintf(SFptr, "%i\n", i);
			   featureIdx.push_back(i);
		   }
	   }
	   cv::Mat temp(featureIdx.size(), trainingDataNum, CV_32FC1);
	   for (int i = 0; i < (int)featureIdx.size(); i++){
		   for (int j = 0; j < trainingDataNum; j++){
			   temp.at<float>(i, j) = genderLbp[stride*j + featureIdx[i]];
		   }
	   }
	   genderLbpMat.push_back(temp);
	   std::cout << "feature size "<<featureIdx.size() << std::endl;
	   delete[] genderLbp;
	   temp.release();
   }
   // [20160108_Sylar]
   //// For svm
   //printf("svm Mat initialize\n");
   //cv::Mat genderLbpMat(cv::Size(selectedFeature.size(), (int)maleImgs.size() + (int)femaleImgs.size()), CV_32FC1);
   //cv::Mat genderLabelMat(cv::Size(1, (int)maleImgs.size() + (int)femaleImgs.size()), CV_32FC1, genderLabel);
   //for (int i = 0; i < (int)selectedFeature.size(); i++){
	  // for (int j = 0; j < trainingDataNum; j++){
		 //  genderLbpMat.at<float>(j, i) = genderLbp[stride*j + selectedFeature[i]];
	  // }
   //}
   //----------
   // set up parameters for SVM
   // [20151124_Curtis] start from a simple model to avoid overfitting
   CvSVMParams params;
   params.svm_type = CvSVM::C_SVC; // C_SVC/NU_SVC/ONE_CLASS/EPS_SVR/NU_SVR
   params.kernel_type = CvSVM::RBF; // LINEAR/POLY/RBF/SIGMOID  
   params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 10e-6);
   
   // start training
   printf("Start training\n");
   cv::Mat gLMt = genderLbpMat.t(); 
   bool ret = m_svm.train_auto(gLMt, genderLabelMat, cv::Mat(), cv::Mat(), params);
   
   // save trained SVM
   m_svm.save(svmModelStr);
   
   // delete
   //delete [] genderLbp;
   delete [] genderLabel;
   
   return ret;
}
bool CLbpSvm::Train(const char* maleImgListStr, const char* femaleImgListStr, const char* svmModelStr)
{
	// --------------------
	// load all male images
	// --------------------
	std::vector<cv::Mat> maleImgs;
	std::vector<cv::Mat> femaleImgs;
	std::ifstream file(maleImgListStr);

	while (!file.eof())
	{
		char faceImgName[256] = { 0 };
		file.getline(faceImgName, 256);
		maleImgs.push_back(cv::imread(faceImgName, 0)); // to grayscale directly
	}
	// load all female images
	file = std::ifstream(femaleImgListStr);
	while (!file.eof())
	{
		char faceImgName[256] = { 0 };
		file.getline(faceImgName, 256);
		femaleImgs.push_back(cv::imread(faceImgName, 0)); // to grayscale directly
	}
	//
	printf("Start training from %d male images and %d female images\n", maleImgs.size(), femaleImgs.size());
	//
	if (maleImgs.size() == 0 || femaleImgs.size() == 0)
	{
		printf("Error: must have enough male/female face images in function CLbpSvm::Train()\n");
		return false;
	}

	// -----------------------------
	// prepare training data for SVM
	// -----------------------------
	// compute LBP
	printf("Convert to LBP\n");
	float* genderLbp = new float[(maleImgs.size() + femaleImgs.size()) * m_blockNum * m_blockNum * 59 * m_scaleMax];
	for (int i = 0; i < maleImgs.size(); i++)
	{
		if (maleImgs[i].empty() == true)
		{
			printf("Error: %i one of male images is incorrect in function CLbpSvm::Train\n", i);
		}

		for (int j = 1; j <= m_scaleMax; j++)
		{
			std::vector<float> lbpHistogram;
			dlib::cv_image<unsigned char> dlibGrayImg(maleImgs[i]);
			dlib::extract_uniform_lbp_descriptors(dlibGrayImg, lbpHistogram, j, 8);
			std::vector<float> lbpHistogramCont = lbpHistogram; // lbpHistogramCont is stored in contiguous memory 
			unsigned int offset = m_blockNum * m_blockNum * 59 * (m_scaleMax * i + j - 1);
			memcpy(&genderLbp[offset], &lbpHistogramCont[0], m_blockNum * m_blockNum * 59 * sizeof(float));

			// (debug)
			if (i <= 3 || i >= maleImgs.size() - 3)
			{
				printf("Male offset = %d\n", offset);
			}
			for (unsigned int k = 0; k < lbpHistogram.size(); k++)
			{
				if (genderLbp[offset + k] != lbpHistogram[k] || genderLbp[offset + k] < 0)
				{
					printf("Error: one of the feature vector is wrong in function CLbpSvm::Train\n");
				}
			}
		}
	}
	for (int i = 0; i < femaleImgs.size(); i++)
	{
		if (femaleImgs[i].empty() == true)
		{
			printf("Error: one of female images is incorrect in function CLbpSvm::Train\n");
		}

		for (int j = 1; j <= m_scaleMax; j++)
		{
			std::vector<float> lbpHistogram;
			dlib::cv_image<unsigned char> dlibGrayImg(femaleImgs[i]);
			dlib::extract_uniform_lbp_descriptors(dlibGrayImg, lbpHistogram, j, 8);
			std::vector<float> lbpHistogramCont = lbpHistogram; // lbpHistogramCont is stored in contiguous memory
			unsigned int offset = m_blockNum * m_blockNum * 59 * (m_scaleMax * ((int)maleImgs.size() + i) + j - 1);
			memcpy(&genderLbp[offset], &lbpHistogramCont[0], m_blockNum * m_blockNum * 59 * sizeof(float)); // lbpHistogram is stored in contiguous memory

			// (debug)
			if (i <= 3 || i >= femaleImgs.size() - 3)
			{
				printf("Female offset = %d\n", offset);
			}
			for (unsigned int k = 0; k < lbpHistogram.size(); k++)
			{
				if (genderLbp[offset + k] != lbpHistogram[k] || genderLbp[offset + k] < 0)
				{
					printf("Error: one of the feature vector is wrong in function CLbpSvm::Train\n");
				}
			}
		}
	}

	// [20151125_Curtis] normalization on the training data and store how to normalize the data
	printf("Start normalization\n");
	unsigned int stride = m_blockNum * m_blockNum * 59 * m_scaleMax;
	FILE* minMaxFptr = fopen("MinMax.csv", "w");
	for (int i = 0; i < m_blockNum * m_blockNum * 59 * m_scaleMax; i++)
	{
		// find min and max
		float min = 99999.0;
		float max = 0.0;
		for (int j = 0; j < maleImgs.size() + femaleImgs.size(); j++)
		{
			float val = genderLbp[stride*j + i];
			if (val < min) min = val;
			if (val > max) max = val;

			// debug
			if (val < 0)
			{
				printf("wrong value in feature vector in function CLbpSvm::Train\n");
			}
		}
		// normalization
		if (min > max)
		{
			printf("Error: cannot have good min/max for noramlization in function CLbpSvm::Train()\n");
			return false;
		}
		else if (min == max)
		{
			min = -1;
			max = 1;
		}
		float normalFactor = (float)2.0 / (max - min);
		for (int j = 0; j < maleImgs.size() + femaleImgs.size(); j++)
		{
			float val = genderLbp[stride*j + i];
			genderLbp[stride*j + i] = (val - min) * normalFactor - 1;
		}
		// store
		fprintf(minMaxFptr, "%.10f %.10f\n", min, max);
	}
	fclose(minMaxFptr);

	// prepare labels for gender
	float* genderLabel = new float[maleImgs.size() + femaleImgs.size()];
	for (unsigned int i = 0; i < maleImgs.size(); i++) // 1 for male
	{
		genderLabel[i] = 1.0;
	}
	for (unsigned int i = 0; i < femaleImgs.size(); i++) // -1 for female
	{
		genderLabel[maleImgs.size() + i] = -1.0;
	}
	//[20160104_Sylar]
	// FeatureSelectionBoost
	//------------------------------------
	const int trainingDataNum = maleImgs.size() + femaleImgs.size(); // the more accurate classifier will be found with more training data 
	cv::Mat trainingData(trainingDataNum, stride, CV_32F); // 32-bit floating
	cv::Mat response(trainingDataNum, 1, CV_32S); // signed integer
	for (int i = 0; i < trainingData.rows; i++)
	{
		for (int j = 0; j < stride; j++){
			trainingData.at<float>(i, j) = genderLbp[stride*i + j];
		}
		// reponse
		response.at<int>(i, 0) = genderLabel[i];
	}
	// training
	cv::Boost Boostmodel;
	printf("Start training boost\n");
	Boostmodel.train(trainingData, CV_ROW_SAMPLE, response, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), CvBoostParams(CvBoost::GENTLE, 100, 0.95, 5, false, 0));
	printf("Done training\n");
	// For Debug
	Boostmodel.save(svmModelStr);
	// Prune
	printf("Find weak predictor\n");
	printf("Number of trees: %d\n", Boostmodel.get_weak_predictors()->total);
	CvSeq* weak = Boostmodel.get_weak_predictors();
	CvSeqReader reader;
	cvStartReadSeq(weak, &reader);
	cvSetSeqReaderPos(&reader, 0);
	std::vector<int> featureVote;
	for (int i = 0; i < weak->total; i++)
	{
		CvBoostTree* wtree;
		CV_READ_SEQ_ELEM(wtree, reader);
		cv::Mat Var = wtree->getVarImportance();
		const CvDTreeNode* node = wtree->get_root();
		CvDTreeSplit* split = node->split;
		const int index = split->condensed_idx;
		featureVote.push_back(index);
		printf("%d\n", index);
	}
	return 1;
	//----------
	// set up parameters for SVM
	// [20151124_Curtis] start from a simple model to avoid overfitting
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC; // C_SVC/NU_SVC/ONE_CLASS/EPS_SVR/NU_SVR
	params.kernel_type = CvSVM::RBF; // LINEAR/POLY/RBF/SIGMOID  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 10e-6);

	// start training
	printf("Start training\n");
	cv::Mat genderLbpMat(cv::Size(m_blockNum * m_blockNum * 59 * m_scaleMax, (int)maleImgs.size() + (int)femaleImgs.size()), CV_32FC1, genderLbp);
	cv::Mat genderLabelMat(cv::Size(1, (int)maleImgs.size() + (int)femaleImgs.size()), CV_32FC1, genderLabel);
	bool ret = m_svm.train_auto(genderLbpMat, genderLabelMat, cv::Mat(), cv::Mat(), params);

	// save trained SVM
	m_svm.save(svmModelStr);

	// delete
	delete[] genderLbp;
	delete[] genderLabel;

	return ret;
}
float CLbpSvm::Predict(const cv::Mat& croppedGrayFaceImg)
{
   if (croppedGrayFaceImg.empty() == true || croppedGrayFaceImg.type() != CV_8UC1)
   {
      printf("Error: input image is not valid or a grayscale face image in function CLbpSvm::Predict()\n");
      return 0;
   }
   // computer LBP
   unsigned int featVecSize = m_blockNum * m_blockNum * 59 * m_scaleMax;
   float* genderLbp = new float[featVecSize];
   for (int j = 1; j <= m_scaleMax; j++)
   {
      std::vector<float> lbpHistogram;
      dlib::cv_image<unsigned char> dlibGrayImg(croppedGrayFaceImg);
      dlib::extract_uniform_lbp_descriptors(dlibGrayImg, lbpHistogram, j, 8);
      std::vector<float> lbpHistogramCont = lbpHistogram; // lbpHistogramCont is stored in contiguous memory
      unsigned int offset = m_blockNum * m_blockNum * 59 * (j - 1);
      memcpy(&genderLbp[offset], &lbpHistogramCont[0], m_blockNum * m_blockNum * 59 * sizeof(float)); // lbpHistogram is stored in contiguous memory
   }
   
   // [20151125_Curtis] normalization
   if (featVecSize != m_min.size() || featVecSize != m_max.size())
   {
      printf("Error: feature vector cannot be normalized in function CLbpSvm::Predict()\n");
      return 0;
   }
   for (unsigned int i = 0; i < featVecSize; i++)
   {
      float normalFactor = (float)2.0 / (m_max[i] - m_min[i]);
      if (genderLbp[i] >= m_max[i]) genderLbp[i] = 1;
      else if (genderLbp[i] <= m_min[i]) genderLbp[i] = -1;
      else genderLbp[i] = (genderLbp[i] - m_min[i]) * normalFactor - 1;
   }
    // [20160106_Sylar]
   // predict
#ifndef FeatureSelection
   cv::Mat testMat(cv::Size(m_blockNum * m_blockNum * 59 * m_scaleMax, 1), CV_32FC1, genderLbp);
#else
   cv::Mat testMat(selectedFeature.size(), 1, CV_32FC1);
   int num = 0 ;
   int scaleCount = 0;
   for (int i = 0; i < (int)selectedFeature.size(); i++){
	   //std::cout << selectedFeature[i] << std::endl;
	   if (num > selectedFeature[i]){
		   scaleCount++;
	   }
	   num = selectedFeature[i];
	   testMat.at<float>(i, 0) = genderLbp[m_blockNum * m_blockNum * 59 * scaleCount + num];
   }
#endif
   
   //[20160104_Sylar] Train data: m_scaleMax = 2
   //float ret = m_boost.predict(testMat);
   float ret = m_svm.predict(testMat, true);
   
   
   delete[] genderLbp;
   return ret;
}
//[20160105_Sylar]
void CLbpSvm::FeatureSelect(float* genderLbp, float* genderLabel, std::vector<int>& featureVote, int scaleNum){
	//[20160104_Sylar]
	// FeatureSelectionBoost
	//------------------------------------
	unsigned int stride = m_blockNum * m_blockNum * 59 ;
	cv::Mat trainingData(trainingDataNum, stride, CV_32F); // 32-bit floating
	cv::Mat response(trainingDataNum, 1, CV_32S); // signed integer
	featureVote.clear();
	featureVote.resize(m_blockNum * m_blockNum * 59 );
	for (int i = 0; i < trainingData.rows; i++)
	{
		for (int j = 0; j < stride; j++){
			trainingData.at<float>(i, j) = genderLbp[stride*i + j];
		}
		// reponse
		response.at<int>(i, 0) = genderLabel[i];
	}
	// training
	cv::Boost Boostmodel;
	printf("Start training boost\n");
	Boostmodel.train(trainingData, CV_ROW_SAMPLE, response, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), CvBoostParams(CvBoost::GENTLE, weakCount, 0.95, 5, surrogate, 0));
	switch (scaleNum){
	case 1:
		Boostmodel.save("nBoost500_scale1.model"); break;
		//Boostmodel.load("Boost500_scale1.model"); break;
	case 2:
		Boostmodel.save("nBoost500_scale2.model");break;
		//Boostmodel.load("Boost500_scale2.model"); break;
	case 3:
		Boostmodel.save("nBoost500_scale3.model");break;
		//Boostmodel.load("Boost500_scale3.model"); break;
	case 4:
		Boostmodel.save("nBoost500_scale4.model");break;
		//Boostmodel.load("Boost500_scale4.model"); break;
	case 5:
		Boostmodel.save("nBoost500_scale5.model");break;
		//Boostmodel.load("Boost500_scale5.model"); break;
	case 6:
		Boostmodel.save("nBoost500_scale6.model");break;
		//Boostmodel.load("Boost500_scale6.model"); break;
	case 7:
		Boostmodel.save("nBoost500_scale7.model");break;
		//Boostmodel.load("Boost500_scale7.model"); break;
	case 8:
		Boostmodel.save("nBoost500_scale8.model");break;
		//Boostmodel.load("Boost500_scale8.model"); break;
	}	
	printf("Done training\n");
	// Prune
	CvSeq* weak = Boostmodel.get_weak_predictors();
	CvSeqReader reader;
	cvStartReadSeq(weak, &reader);
	cvSetSeqReaderPos(&reader, 0);
	printf("weak->total %i\n", weak->total);
	std::vector<double> importance;
	importance.resize(stride);
	for (int i = 0; i < weak->total; i++)
	{
		CvBoostTree* wtree;
		CV_READ_SEQ_ELEM(wtree, reader);
		const CvDTreeNode* node = wtree->get_root();
		CvDTreeSplit* split = node->split;
		int index = split->var_idx;		
		// [20160108_Sylar] Feature selected by Variable Importance
		cv::Mat VarImp = wtree->getVarImportance();
		int cols = VarImp.cols;
		double max_importance = 0;
		int importance_idx=0;
		//std::cout << VarImp.row(0) << std::endl;
		for (int j = 0; j < cols; j++){
			/*if (VarImp.at<double>(0, j) * 10000 > max_importance){
				max_importance = VarImp.at<double>(0, j) * 10000;
				importance_idx = j;
			}*/
			importance[j] += VarImp.at<double>(0, j);
		}
		//featureVote[index]++;
		//featureVote[importance_idx]++;
		//printf("index %d, importance %lf\n", index, importance);
	}
	// [20160115_Sylar]
	std::map<double, int, std::greater<double>> maxFeatures;
	for (int i = 0; i < (int)importance.size(); i++){
		maxFeatures.insert(std::make_pair(importance[i], i));
	}
	int count = 0;
	for (std::map<double, int>::iterator it = maxFeatures.begin(); it != maxFeatures.end(); it++){
		if (count >= 500)
			break;
		featureVote[it->second]++;
		count++;
		std::cout << it->second << " ";
	}
	// [20160108_Sylar] Debug
	/*double count = 0;
	for (int i = 0; i < (int)importance.size(); i++){		
		if (i % 59 == 0 && i!=0){
			std::cout << count << " ";
			count = 0;
		}		
		count += importance[i] ;
		if (i % 944 == 0 && i!=0)
			std::cout << std::endl;
	}
	std::cout << count << std::endl;*/
	return ;
}
//[20160105_Sylar]
bool CLbpSvm::ComputeLbp(float* genderLbp, const std::vector<cv::Mat>& maleImgs, const std::vector<cv::Mat>& femaleImgs, int scaleNum){
	// -----------------------------
	// prepare training data for SVM
	// -----------------------------
	// compute LBP
	printf("Convert to LBP\n");
	for (int i = 0; i < maleImgs.size(); i++)
	{
		if (maleImgs[i].empty() == true)
		{
			printf("Error: %i one of male images is incorrect in function CLbpSvm::Train\n", i);
		}

			std::vector<float> lbpHistogram;
			dlib::cv_image<unsigned char> dlibGrayImg(maleImgs[i]);
			dlib::extract_uniform_lbp_descriptors(dlibGrayImg, lbpHistogram, scaleNum, (128/m_blockNum) );
			std::vector<float> lbpHistogramCont = lbpHistogram; // lbpHistogramCont is stored in contiguous memory 
			unsigned int offset = m_blockNum * m_blockNum * 59 * i ;
			memcpy(&genderLbp[offset], &lbpHistogramCont[0], m_blockNum * m_blockNum * 59 * sizeof(float));

		
	}
	for (int i = 0; i < femaleImgs.size(); i++)
	{
		if (femaleImgs[i].empty() == true)
		{
			printf("Error: one of female images is incorrect in function CLbpSvm::Train\n");
		}

			std::vector<float> lbpHistogram;
			dlib::cv_image<unsigned char> dlibGrayImg(femaleImgs[i]);
			dlib::extract_uniform_lbp_descriptors(dlibGrayImg, lbpHistogram, scaleNum, (128 / m_blockNum) );
			std::vector<float> lbpHistogramCont = lbpHistogram; // lbpHistogramCont is stored in contiguous memory
			unsigned int offset = m_blockNum * m_blockNum * 59 * ( (int)maleImgs.size() + i );
			memcpy(&genderLbp[offset], &lbpHistogramCont[0], m_blockNum * m_blockNum * 59 * sizeof(float)); // lbpHistogram is stored in contiguous memory
					
	}

	// [20151125_Curtis] normalization on the training data and store how to normalize the data
	printf("Start normalization\n");
	unsigned int stride = m_blockNum * m_blockNum * 59 ;
	FILE* minMaxFptr = fopen("MinMax.csv", "a");
	for (int i = 0; i < m_blockNum * m_blockNum * 59 ; i++)
	{
		// find min and max
		float min = 99999.0;
		float max = 0.0;
		for (int j = 0; j < maleImgs.size() + femaleImgs.size(); j++)
		{
			float val = genderLbp[stride*j + i];
			if (val < min) min = val;
			if (val > max) max = val;

			// debug
			if (val < 0)
			{
				printf("wrong value in feature vector in function CLbpSvm::Train\n");
			}
		}
		// normalization
		if (min > max)
		{
			printf("Error: cannot have good min/max for noramlization in function CLbpSvm::Train()\n");
			return false;
		}
		else if (min == max)
		{
			min = -1;
			max = 1;
		}
		float normalFactor = (float)2.0 / (max - min);
		for (int j = 0; j < maleImgs.size() + femaleImgs.size(); j++)
		{
			float val = genderLbp[stride*j + i];
			genderLbp[stride*j + i] = (val - min) * normalFactor - 1;
		}
		// store
		//if (scaleNum == 9)
			fprintf(minMaxFptr, "%.10f %.10f\n", min, max);
	
	}
	fclose(minMaxFptr);
	return true;
}
