#include "SSD.h"

#include <iosfwd>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#if defined(_MSC_VER)
#include <io.h>
#endif
#include <fcntl.h>

using namespace caffe;
using std::string;

SSD::SSD(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& mean_value,
                       const string& label_file,
                       GPUAllocator* allocator,
					   float iou_threshold,
					   float ios_threshold,
					   float conf_threshold)
    : allocator_(allocator),
	  iou_(iou_threshold),
	  ios_(ios_threshold),
	  conf_(conf_threshold)
{
	LOG(INFO) << iou_ << " " << ios_ << " " << conf_;
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_ = std::make_shared<Net<float>>(model_file, TEST);
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);

    /* Load labels. */
	LabelMap labelMap;
#if defined (_MSC_VER)  // for MSC compiler binary flag needs to be specified
	int fd;
	_sopen_s(&fd, label_file.c_str(), O_RDONLY | O_BINARY, _SH_DENYNO, 0);
#else
	int fd = open(filename, O_RDONLY);
#endif
	CHECK_NE(fd, -1) << "Unable to open labels file " << label_file;
	google::protobuf::io::FileInputStream fileInput(fd);
	fileInput.SetCloseOnDelete(true);
	CHECK(google::protobuf::TextFormat::Parse(&fileInput, &labelMap)) << "Parse failed from: " << label_file;
	
	for (int i = 0; i < labelMap.item_size(); ++i)
	{
		const LabelMapItem& item = labelMap.item(i);
		labels_.push_back(item.display_name());
	}

	////////////////////////////////
	//Retrive labels
	////////////////////////////////
	/*if (!labels_.empty())
	{
		std::ofstream ostream("labels.txt");
		CHECK(ostream.is_open()) << "labels.txt isn't open";
		for (size_t i = 0; i < labels_.size(); ++i)
		{
			ostream << i << " " << labels_[i] << std::endl;
		}
		ostream.close();
	}*/

    CHECK_EQ(labels_.size(), net_->layer_by_name("detection_out")->layer_param().detection_output_param().num_classes())
        << "Number of labels is different from the output layer's parameter.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector< std::vector<float> >& v, int N)
{
    std::vector< std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i][1], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
DetectedBBoxes SSD::Detect(const Mat& img, int N)
{
	Size frameSize(img.cols, img.rows);
    std::vector< std::vector<float> > output = Predict(img);

	N = N <= 0 ? output.size() : std::min<int>(output.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    DetectedBBoxes detectedBoxes;
    for (int i = 0; i < N; ++i)
    {
        int idx = maxN[i];
		detectedBoxes.emplace_back(output[idx], labels_[cvRound(output[idx][1])], frameSize);
    }

    return detectedBoxes;
}

Mat SSD::DetectAsMat(const Mat& img)
{
	std::vector< std::vector<float> > output = Predict(img);
	cv::Mat detections(output.size(), 7, CV_32F);

	int r = 0;
	for (auto objInfo : output)
	{
		cv::Mat obj(1, objInfo.size(), CV_32F, objInfo.data());
		obj.copyTo(detections.row(r));
		r++;
	}

	return detections;
}

const std::vector<std::string>& SSD::GetLabels() const
{
	return labels_;
}

const std::string& SSD::GetLabel(size_t idx) const
{
	return labels_[idx];
}

std::vector< std::vector<float> > SSD::Predict(const Mat& img)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<GpuMat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* result = output_layer->cpu_data();
    const int num_det = output_layer->height();
    std::vector< std::vector<float> > detections;
    for(int k = 0; k < num_det; ++k)
    {
        if(result[1] < conf_)
        {
            //Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }

    return detections;
}

/* Load the mean file in binaryproto format. */
void SSD::SetMean(const string& mean_file, const string& mean_value)
{
    Scalar channel_mean;
    if (!mean_file.empty())
    {
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<Mat> channels;
        float *data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        Mat packed_mean;
        merge(channels, packed_mean);

        /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
        channel_mean = mean(packed_mean);
        Mat host_mean = Mat(input_geometry_, packed_mean.type(), channel_mean);
        mean_.upload(host_mean);
    }
    if (!mean_value.empty())
    {
        CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ','))
        {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) << "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i)
        {
            /* Extract an individual channel. */
            Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, Scalar(values[i]));
            channels.push_back(channel);
        }

        Mat packed_mean;
        merge(channels, packed_mean);

        mean_.upload(packed_mean);
    }
}

/* Wrap the input layer of the network in separate GpuMat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SSD::WrapInputLayer(std::vector<GpuMat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_gpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        GpuMat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void SSD::Preprocess(const Mat& host_img,
                            std::vector<GpuMat>* input_channels)
{
    GpuMat img(host_img, allocator_);
    /* Convert the input image to the input image format of the network. */
    GpuMat sample(allocator_);
    if (img.channels() == 3 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    GpuMat sample_resized(allocator_);
    if (sample.size() != input_geometry_)
        cuda::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    GpuMat sample_float(allocator_);
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    GpuMat sample_normalized(allocator_);
    cuda::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the GpuMat
     * objects in input_channels. */
    cuda::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->gpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}