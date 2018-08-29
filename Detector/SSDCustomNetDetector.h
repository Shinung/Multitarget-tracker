#pragma once

#include "BaseDetector.h"

/**
 * @brief Detector class for custom caffe-ssd net
 * 
 * This class was implemented to refer NVIDIA Gpu Rest Engine code
 * 
 * @see [forked GRE branch](https://github.com/Shinung/gpu-rest-engine/tree/ssd_gre/caffe_ssd)
 * @author Shinung
 */
class SSDCustomNetDetector : public BaseDetector
{
	typedef struct CustomSSD_ctx CustomSSD_ctx;

public:
	SSDCustomNetDetector(bool collectPoints, cv::UMat& colorFrame);
	~SSDCustomNetDetector();

	/**
	 * @brief Initialize Detector and SSD net
	 * 
	 * This function was implemented to refer NVIDIA GRE code.
	 * 
	 * @param config Setted values in CustomSSDMobileNetExample::InitTracker
	 * @see Reference [code](https://github.com/Shinung/gpu-rest-engine/blob/ssd_gre/caffe_ssd/detection.cpp)
	 * @author Shinung
	 */
	virtual bool Init(const config_t& config) override;

	/**
	 * @brief Detecting with SSD net
	 * 
	 * Forwarding SSD net with cropped gray images.
	 * 
	 * @param [in] colorFrame Always input gray scale image in this parameter
	 * 			   because CustomSSDMobileNetExample::GrayProcessing return only TRUE
	 */
	virtual void Detect(cv::UMat& colorFrame) override;

private:

	/**
	 * @brief Making cropped images and fowarding SSD net
	 * 
	 * @param [in]  colorFrame
	 * @param [in]  crop
	 * @param [out] tmpRegions
	 * @todo It can be bottleneck with big image
	 * 		 I think to improve process time if each cropped image forward to SSD net,
	 * 		 which live in context pool
	 * @see [detection_inference function - 149 line](https://github.com/Shinung/gpu-rest-engine/blob/ssd_gre/caffe_ssd/detection.cpp)
	 * @author Shinung
	 */
	void DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions);

private:
	/**
	 * @brief A struct of ContextPool
	 * 
	 * I just followed [NVIDIA GRE](https://github.com/NVIDIA/gpu-rest-engine) codes
	 * although It's not necessary for multi-threads logic in here.
	 * 
	 * @see common.h
	 * @author Shinung
	 */
	CustomSSD_ctx* mCTX;

	int InWidth;
	int InHeight;
	float m_WHRatio;
	float m_inScaleFactor;
	float m_confidenceThreshold;
	float m_maxCropRatio;
	std::vector<std::string> m_classNames; /**< Container for objects label */
};