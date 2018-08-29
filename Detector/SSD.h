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

/**
 * @brief Bounding Box class
 * 
 * Saving a object's information from SSD
 * @author Shinung
 */
class BBox
{
public:

	BBox() { }

	/**
	 * Constructor to parse a information from SSD
	 * 
	 * 0        | 1        | 2        | 3        | 4        | 5        | 6
	 * image_id | label    | score    | xmin     | ymin     | xmax     | ymax
	 * 
	 * @param info A object's information
	 * @param label A object label
	 * @param frameSize image size for getting top-left and bottom-right coordinate with information
	 * @author Shinung
	 */
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
    Rect mBox; /**< A rect of object */
	Point mCenter; /**< Center coord of rect */
    float mConf; /**< Confidence(score) */
	std::pair<int, std::string> mLabel; /**< Object label */
};

/**
 * Container for BBox
 * @author Shinung
 */
typedef std::vector<BBox> DetectedBBoxes;

class SSD
{
public:

	/**
	 * @brief SSD Constructor
	 * @param model_file deploy prototxt path
	 * @param trained_file caffemodel prototxt path
	 * @param mean_file "" or mean binaryproto file path
	 * @param mean_value "104,117,123" - Use this parameter if not having a mean file
	 * @param label_file labelmap prototxt path
	 * @param allocator Instance of GPUAllocator
	 * @param iou_threshold IOU threshold value
	 * @param ios_threshold IOS threshold value
	 * @param conf_threshold Confidence threshold value
	 */
    SSD(const std::string& model_file,
               const std::string& trained_file,
               const std::string& mean_file,
               const std::string& mean_value,
               const std::string& label_file,
               GPUAllocator* allocator,
			   float conf_threshold);

	/**
	 * @brief Forwarding SSD with a image
	 * 
	 * This function is not used with MultiTarget-Tracker due to return type
	 * The reason why I didn't overloading is to follow [popetv's coding standards](https://docs.google.com/document/d/1cT8EPgMXe0eopeHvwuFmbHG4TJr5kUmcovkr5irQZmo/edit#heading=h.r5zmbif4dbnf)
	 * I love his coding philosophy personally
	 * 
	 * @since [forked GRE/SSD.h](https://github.com/Shinung/gpu-rest-engine/blob/ssd_gre/caffe_ssd/SSD.h)
	 * @param img Decoding image to cv::Mat
	 * @param N Limit number of detecting objects
	 * @return DetectedBBoxes
	 * @author Shinung
	 */
    virtual DetectedBBoxes Detect(const Mat& img, int N = 5);

	/**
	 * @brief Forwarding SSD with a image
	 * @param img Decoding image to cv::Mat
	 * @return cv::Mat that has raw results from SSD instead of DetectedBBoxes
	 * @author Shinung
	 */
	virtual Mat DetectAsMat(const Mat& img);
	
	/**
	 * @brief Return all labels
	 * @return std::vector of labels
	 * @author Shinung
	 */
	const std::vector<std::string>& GetLabels() const;

	/**
	 * @brief Return a label by index
	 * @return label string
	 * @author Shinung
	 */
	const std::string& GetLabel(size_t idx) const;

	/**
	 * @return Size of input layer of SSD
	 * @author Shinung
	 */
	const cv::Size& GetInputGeometry() const;

protected:
    std::vector< std::vector<float> > Predict(const Mat& img);

private:
    void SetMean(const std::string& mean_file, const std::string& mean_value);

    void WrapInputLayer(std::vector<GpuMat>* input_channels);

    void Preprocess(const Mat& img,
                    std::vector<GpuMat>* input_channels);

private:

	/**
	 * @see gpu_allocator.h
	 * @author Shinung
	 */
    GPUAllocator* allocator_;
    std::shared_ptr<caffe::Net<float>> net_;
    Size input_geometry_;
    int num_channels_;
    GpuMat mean_;
    std::vector<std::string> labels_;

	float conf_;
};

#endif