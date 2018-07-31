#pragma once

#include "BaseDetector.h"
///
/// \brief The SSDMobileNetDetector class
///
class SSDCustomNetDetector : public BaseDetector
{
	typedef struct CustomSSD_ctx CustomSSD_ctx;

public:
	SSDCustomNetDetector(bool collectPoints, cv::UMat& colorFrame);
	~SSDCustomNetDetector(void);

	bool Init(const config_t& config);

	void Detect(cv::UMat& colorFrame);

private:
	void DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions);

private:
	CustomSSD_ctx* mCTX;
	cv::dnn::Net m_net;

	int InWidth;
	int InHeight;
	float m_WHRatio;
	float m_inScaleFactor;
	float m_meanVal;
	float m_confidenceThreshold;
	float m_maxCropRatio;
	std::vector<std::string> m_classNames;
};