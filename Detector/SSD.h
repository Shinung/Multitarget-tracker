#ifndef SSD_H
#define SSD_H

#include <string>
#include <vector>
#include <limits>
#include <cmath>

#define USE_CUDNN 1
#include <caffe/caffe.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "gpu_allocator.h"

class BBox
{
public:
	BBox() { }
	BBox(const std::vector<float>& info, const std::string& label, const Size& frameSize)
	{
		mBox = Rect(Point(cvRound(info[3] * frameSize.width), cvRound(info[4] * frameSize.height)),
			Point(cvRound(info[5] * frameSize.width), cvRound(info[6] * frameSize.height)));

		mCenter = Point(cvRound((mBox.br().x + mBox.tl().x) / 2.f), cvRound((mBox.br().y + mBox.tl().y) / 2.f));

		mConf = info[2];

		mLabel = std::make_pair(cvRound(info[1]), label);
	}
	BBox(const BBox& other)
	{
		mBox = other.mBox;
		mCenter = other.mCenter;
		mConf = other.mConf;
		mLabel = other.mLabel;
	}
	BBox(const BBox&& rhs)
	{
		mBox = rhs.mBox;
		mCenter = rhs.mCenter;
		mConf = rhs.mConf;
		mLabel = rhs.mLabel;
	}

	BBox& operator=(const BBox& rhs)
	{
		if (this == &rhs)
		{
			return *this;
		}

		mBox = rhs.mBox;
		mCenter = rhs.mCenter;
		mConf = rhs.mConf;
		mLabel = rhs.mLabel;

		return *this;
	}

public:
    Rect mBox;
	Point mCenter;
    float mConf;
	std::pair<int, std::string> mLabel;
};

typedef std::vector<BBox> DetectedBBoxes;

class SSD
{
public:
    SSD(const std::string& model_file,
               const std::string& trained_file,
               const std::string& mean_file,
               const std::string& mean_value,
               const std::string& label_file,
               GPUAllocator* allocator,
			   float iou_threshold,
			   float ios_threshold,
			   float conf_threshold);

    virtual DetectedBBoxes Detect(const Mat& img, int N = 5);
	virtual Mat DetectAsMat(const Mat& img);
	
	const std::vector<std::string>& GetLabels() const;
	const std::string& GetLabel(size_t idx) const;

protected:
    std::vector< std::vector<float> > Predict(const Mat& img);

private:
    void SetMean(const std::string& mean_file, const std::string& mean_value);

    void WrapInputLayer(std::vector<GpuMat>* input_channels);

    void Preprocess(const Mat& img,
                    std::vector<GpuMat>* input_channels);

private:
    GPUAllocator* allocator_;
    std::shared_ptr<caffe::Net<float>> net_;
    Size input_geometry_;
    int num_channels_;
    GpuMat mean_;
    std::vector<std::string> labels_;

	float iou_, ios_, conf_;
};

#endif