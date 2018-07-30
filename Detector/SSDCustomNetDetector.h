#pragma once

#include "BaseDetector.h"

///
/// \brief The SSDMobileNetDetector class
///
class SSDCustomNetDetector : public BaseDetector
{
public:
	SSDCustomNetDetector(bool collectPoints, cv::UMat& colorFrame);
	~SSDCustomNetDetector(void);

	bool Init(const config_t& config);

	void Detect(cv::UMat& colorFrame);

private:
	cv::dnn::Net m_net;

	void DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions);

	static const int InWidth = 300;
	static const int InHeight = 300;
	float m_WHRatio;
	float m_inScaleFactor;
	float m_meanVal;
	float m_confidenceThreshold;
	float m_maxCropRatio;
	std::vector<std::string> m_classNames;
};