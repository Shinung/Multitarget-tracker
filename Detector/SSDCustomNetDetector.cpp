#include "SSDCustomNetDetector.h"
#include "nms.h"
#include "common.h"
#include "SSD.h"

//#define NONCROP

using namespace caffe;
using std::string;

/**
 * By using Go as the HTTP server, we have potentially more CPU threads than
 * available GPUs and more threads can be added on the fly by the Go
 * runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
 * when a CPU thread is ready for inference it will try to retrieve an
 * execution context from a queue of available GPU contexts and then do a
 * cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
 * per GPU. - From [NVIDIA/gpu-rest-engine](https://github.com/NVIDIA/gpu-rest-engine)
 * 
 * @see <https://github.com/NVIDIA/gpu-rest-engine>
 * @see forked: <https://github.com/Shinung/gpu-rest-engine/tree/ssd_gre>
 * @author Shinung
 */
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

/**
 * Currently, 2 execution contexts are created per GPU. In other words, 2
 * inference tasks can execute in parallel on the same GPU. This helps improve
 * GPU utilization since some kernel operations of inference will not fully use
 * the GPU. From [NVIDIA/gpu-rest-engine](https://github.com/NVIDIA/gpu-rest-engine)
 * 
 * @author Shinung
 */
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

bool SSDCustomNetDetector::Init(const config_t& config)
{
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

	try
	{
		int device_count;
		cudaError_t st = cudaGetDeviceCount(&device_count);
		if (st != cudaSuccess)
		{
			throw std::invalid_argument("could not list CUDA devices");
		}

		ContextPool<ExecContext> pool;
		for (int dev = 0; dev < device_count; ++dev)
		{
			if (!ExecContext::IsCompatible(dev))
			{
				LOG(ERROR) << "Skipping device: " << dev;
				continue;
			}

			for (int i = 0; i < kContextsPerDevice; ++i)
			{
				std::unique_ptr<ExecContext> context(
						new ExecContext(
								config.find("modelConfiguration")->second,
								config.find("modelBinary")->second,
								"",
								config.find("meanValue")->second,
								config.find("labelMap")->second,
								m_confidenceThreshold,
								dev
						)
				);
				pool.Push(std::move(context));
			}
		}

		if (pool.Size() == 0)
		{
			throw std::invalid_argument("no suitable CUDA device");
		}
		mCTX = new CustomSSD_ctx{ std::move(pool) };

		{
			ScopedContext<ExecContext> context(mCTX->pool);
			SSD* ssd = static_cast<SSD*>(context->CaffeClassifier());
			InWidth = ssd->GetInputGeometry().width;
			InHeight = ssd->GetInputGeometry().height;
		}

		/* Successful CUDA calls can set errno. */
		errno = 0;
		return true;
	}
	catch (const std::invalid_argument& ex)
	{
		LOG(ERROR) << "exception: " << ex.what();
		errno = EINVAL;
		return false;
	}
}

void SSDCustomNetDetector::Detect(cv::UMat& colorFrame)
{
	m_regions.clear();

	regions_t tmpRegions;

	cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);

#ifdef NONCROP
	cv::Rect crop(0, 0, colorMat.cols, colorMat.rows);
    DetectInCrop(colorMat, crop, tmpRegions);
#else
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
#endif

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
	// get ssd instance from the pool
    ScopedContext<ExecContext> context(mCTX->pool);
    SSD* ssd = static_cast<SSD*>(context->CaffeClassifier());

    cv::Mat detectionMat = ssd->DetectAsMat(cv::Mat(colorFrame, crop));

	for (int i = 0; i < detectionMat.rows; ++i)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > m_confidenceThreshold)
		{
			size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

#ifdef NONCROP
            int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * crop.width);
            int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * crop.height);
            int xRightTop = cvRound(detectionMat.at<float>(i, 5) * crop.width);
            int yRightTop = cvRound(detectionMat.at<float>(i, 6) * crop.height);
#else
            int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * crop.width) + crop.x;
			int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * crop.height) + crop.y;
			int xRightTop = cvRound(detectionMat.at<float>(i, 5) * crop.width) + crop.x;
			int yRightTop = cvRound(detectionMat.at<float>(i, 6) * crop.height) + crop.y;

			if (xLeftBottom < 0 || yLeftBottom < 0 || xRightTop < 0 || yRightTop < 0)
			{
				continue;
			}
#endif
			CHECK_LE(0, xLeftBottom) << "raw: " << detectionMat.at<float>(i, 3);
			CHECK_LE(0, yLeftBottom) << "raw: " << detectionMat.at<float>(i, 4);
			CHECK_LE(0, xRightTop) << "raw: " << detectionMat.at<float>(i, 5);
			CHECK_LE(0, yRightTop) << "raw: " << detectionMat.at<float>(i, 6);

			cv::Rect object(cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop));

			tmpRegions.push_back(CRegion(object, ssd->GetLabel(objectClass), confidence));
		}
	}
}

