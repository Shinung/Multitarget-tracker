#include "SSDCustomNetDetector.h"
#include "nms.h"
#include "common.h"
#include "SSD.h"

using namespace caffe;
using std::string;

/* By using Go as the HTTP server, we have potentially more CPU threads than
* available GPUs and more threads can be added on the fly by the Go
* runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
* when a CPU thread is ready for inference it will try to retrieve an
* execution context from a queue of available GPU contexts and then do a
* cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
* per GPU. */
class ExecContext
{
public:
	friend ScopedContext<ExecContext>;

	static bool IsCompatible(int device)
	{
		cudaError_t st = cudaSetDevice(device);
		if (st != cudaSuccess)
			return false;

		cuda::DeviceInfo info;
		if (!info.isCompatible())
			return false;

		return true;
	}

	ExecContext(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& mean_value,
		const string& label_file,
		float conf_threshold,
		int device)
		: device_(device)
	{
		cudaError_t st = cudaSetDevice(device_);
		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");

		allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
		caffe_context_.reset(new Caffe);
		Caffe::Set(caffe_context_.get());
		detector_.reset(new SSD(model_file, trained_file,
			mean_file, mean_value,
			label_file,
			allocator_.get(),
			conf_threshold));

		Caffe::Set(nullptr);
	}

	SSD* CaffeClassifier()
	{
		return detector_.get();
	}

private:
	void Activate()
	{
		cudaError_t st = cudaSetDevice(device_);
		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");
		allocator_->reset();
		Caffe::Set(caffe_context_.get());
	}

	void Deactivate()
	{
		Caffe::Set(nullptr);
	}

private:
	int device_;
	std::unique_ptr<GPUAllocator> allocator_;
	std::unique_ptr<Caffe> caffe_context_;
	std::unique_ptr<SSD> detector_;
};

struct CustomSSD_ctx
{
	ContextPool<ExecContext> pool;
};

/* Currently, 2 execution contexts are created per GPU. In other words, 2
* inference tasks can execute in parallel on the same GPU. This helps improve
* GPU utilization since some kernel operations of inference will not fully use
* the GPU. */
constexpr static int kContextsPerDevice = 2;

/// \brief SSDCustomNetDetector::SSDMobileNetDetector
/// \param collectPoints
/// \param gray
///
SSDCustomNetDetector::SSDCustomNetDetector(
	bool collectPoints,
	cv::UMat& colorFrame
)
	:
	BaseDetector(collectPoints, colorFrame),
	m_WHRatio(1.f),
	m_inScaleFactor(0.007843f),
	m_meanVal(127.5f),
	m_confidenceThreshold(0.5f),
	m_maxCropRatio(2.0f)
{
}

///
/// \brief SSDMobileNetDetector::~SSDMobileNetDetector
///
SSDCustomNetDetector::~SSDCustomNetDetector(void)
{
	delete mCTX;
}

///
/// \brief SSDCustomNetDetector::Init
/// \return
///
bool SSDCustomNetDetector::Init(const config_t& config)
{
	/* Comment by Roy
	 * Initialize caffe ssd net
	 * Refer classifier_initialize function
	 * in https://github.com/NVIDIA/gpu-rest-engine/blob/master/caffe/classification.cpp
	 * and check SSD class's constructor how to initialize caffe model.
	 */


	auto modelConfiguration = config.find("modelConfiguration");
	if (modelConfiguration != config.end())
	{
		model_file = modelConfiguration->second;
	}

	auto modelBinary = config.find("modelBinary");
	if (modelBinary != config.end())
	{
		trained_file = modelBinary->second;
	}

//	if (modelConfiguration != config.end() && modelBinary != config.end())
//	{
//		m_net = cv::dnn::readNetFromCaffe(modelConfiguration->second, modelBinary->second);
//	}

	auto labelMap = config.find("labelMap");

	auto meanValue = config.find("meanValue");
	if (meanValue != config.end())
	{
		m_meanVal = std::stof(meanValue->second);
	}

	auto confidenceThreshold = config.find("confidenceThreshold");
	if (confidenceThreshold != config.end())
	{
		m_confidenceThreshold = std::stof(confidenceThreshold->second);
	}

	auto maxCropRatio = config.find("maxCropRatio");
	if (maxCropRatio != config.end())
	{
		m_maxCropRatio = std::stof(maxCropRatio->second);
		if (m_maxCropRatio < 1.f)
		{
			m_maxCropRatio = 1.f;
		}
	}

	// TODO: ExecContext pool
	// TODO: get size from SSD
	

	return !m_net.empty();
}

///
/// \brief SSDCustomNetDetector::Detect
/// \param gray
///
void SSDCustomNetDetector::Detect(cv::UMat& colorFrame)
{
	/* Comment by Roy
	 * Here for forwarding SSD net to detect objects
	 * Should be combine well between logic in this method and 'Detect' method in our SSD class
	 * This logic is assumed that a frame is cropped by crop ratio and each cropped images will input to SSD net.
	 * Therefore, our trained net(SSD 512x512) was slow. It would have been bottleneck.
	 * I hope to improve fps if we'll combine successfully.
	 */
	m_regions.clear();

	regions_t tmpRegions;

	cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);

	int cropHeight = cvRound(m_maxCropRatio * InHeight);
	int cropWidth = cvRound(m_maxCropRatio * InWidth);

	if (colorFrame.cols / (float)colorFrame.rows > m_WHRatio)
	{
		if (m_maxCropRatio <= 0 || cropHeight >= colorFrame.rows)
		{
			cropHeight = colorFrame.rows;
		}
		cropWidth = cvRound(cropHeight * m_WHRatio);
	}
	else
	{
		if (m_maxCropRatio <= 0 || cropWidth >= colorFrame.cols)
		{
			cropWidth = colorFrame.cols;
		}
		cropHeight = cvRound(colorFrame.cols / m_WHRatio);
	}

	cv::Rect crop(0, 0, cropWidth, cropHeight);

	for (; crop.y < colorMat.rows; crop.y += crop.height / 2)
	{
		bool needBreakY = false;
		if (crop.y + crop.height >= colorMat.rows)
		{
			crop.y = colorMat.rows - crop.height;
			needBreakY = true;
		}
		for (crop.x = 0; crop.x < colorMat.cols; crop.x += crop.width / 2)
		{
			bool needBreakX = false;
			if (crop.x + crop.width >= colorMat.cols)
			{
				crop.x = colorMat.cols - crop.width;
				needBreakX = true;
			}

			DetectInCrop(colorMat, crop, tmpRegions);

			if (needBreakX)
			{
				break;
			}
		}
		if (needBreakY)
		{
			break;
		}
	}

	nms3<CRegion>(tmpRegions, m_regions, 0.4f,
		[](const CRegion& reg) -> cv::Rect { return reg.m_rect; },
		[](const CRegion& reg) -> float { return reg.m_confidence; },
		[](const CRegion& reg) -> std::string { return reg.m_type; },
		0, 0.f);

	if (m_collectPoints)
	{
		for (auto& region : m_regions)
		{
			CollectPoints(region);
		}
	}
}

///
/// \brief SSDCustomNetDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void SSDCustomNetDetector::DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
	/* Comment by Roy
	 * Here is ACTUALLY the phase of forwarding SSD net with cropped images.
	 * I wouldn't like to remove this code because there are some different way to make results from net-output
	 * with our SSD class.
	 * our SSD class will make BBox from each detected object but, on the other hand, this code will make CRegion.
	 * Hence, we should make CRegion from our SSD class to integrate with this code.
	 */

	//Convert Mat to batch of images
	cv::Mat inputBlob = cv::dnn::blobFromImage(cv::Mat(colorFrame, crop), m_inScaleFactor, cv::Size(InWidth, InHeight), m_meanVal, false, true);

	m_net.setInput(inputBlob, "data"); //set the network input

	cv::Mat detection = m_net.forward("detection_out"); //compute output

														//std::vector<double> layersTimings;
														//double freq = cv::getTickFrequency() / 1000;
														//double time = m_net.getPerfProfile(layersTimings) / freq;

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	//cv::Mat frame = colorFrame(crop);

	//ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
	//putText(frame, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));
	//std::cout << "Inference time, ms: " << time << endl;

	//cv::Point correctPoint((colorFrame.cols - crop.width) / 2, (colorFrame.rows - crop.height) / 2);

	for (int i = 0; i < detectionMat.rows; ++i)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > m_confidenceThreshold)
		{
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

			int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * crop.width) + crop.x;
			int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * crop.height) + crop.y;
			int xRightTop = cvRound(detectionMat.at<float>(i, 5) * crop.width) + crop.x;
			int yRightTop = cvRound(detectionMat.at<float>(i, 6) * crop.height) + crop.y;

			cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

			tmpRegions.push_back(CRegion(object, m_classNames[objectClass], confidence));

			//cv::rectangle(frame, object, Scalar(0, 255, 0));
			//std::string label = classNames[objectClass] + ": " + std::to_string(confidence);
			//int baseLine = 0;
			//cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			//cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
			//cv::putText(frame, label, Point(xLeftBottom, yLeftBottom), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
		}
	}
}

