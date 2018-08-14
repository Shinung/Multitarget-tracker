#pragma once

#include "BaseDetector.h"

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include "nms.h"

// ----------------------------------------------------------------------

///
/// \brief Initialization GLOG library
///
void GlobalInitGLOG(char*** pargv);

// ----------------------------------------------------------------------

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser);
    virtual ~VideoExample();

    void Process();

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    bool m_showLogs;
    float m_fps;
    bool m_useLocalTracking;

    int m_captureTimeOut;
    int m_trackingTimeOut;

    static void CaptureAndDetect(VideoExample* thisPtr,
                                 bool* stopCapture,
                                 std::mutex* frameLock,
                                 std::condition_variable* frameCond,
                                 std::mutex* trackLock,
                                 std::condition_variable* trackCond);

    virtual bool GrayProcessing() const;

    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions);
    void Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions);

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track,
                   bool drawTrajectory = true,
                   bool isStatic = false
                   );
private:
    bool m_isTrackerInitialized;
    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame;
    int m_endFrame;
    int m_finishDelay;
    std::vector<cv::Scalar> m_colors;

    struct FrameInfo
    {
        cv::Mat m_frame;
        cv::UMat m_gray;
        regions_t m_regions;
        int64 m_dt;

        FrameInfo()
            : m_dt(0)
        {

        }
    };
    FrameInfo m_frameInfo[2];

    int m_currFrame;
};

// ----------------------------------------------------------------------

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser),
          m_minObjWidth(10)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        m_useLocalTracking = false;

        m_minObjWidth = frame.cols / 50;

        const int minStaticTime = 5;

        config_t config;
#if 0
        config["history"] = std::to_string(cvRound(10 * minStaticTime * m_fps));
        config["varThreshold"] = "16";
        config["detectShadows"] = "1";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, m_useLocalTracking, frame));
#else
        config["minPixelStability"] = "15";
        config["maxPixelStability"] = "900";
        config["useHistory"] = "1";
        config["isParallel"] = "1";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, m_useLocalTracking, frame));
#endif
        m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistCenters;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;    // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.5f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 20;           // Distance threshold between region and object on two frames

        settings.m_useAbandonedDetection = true;
        if (settings.m_useAbandonedDetection)
        {
            settings.m_minStaticTime = minStaticTime;
            settings.m_maxStaticTime = 60;
            settings.m_maximumAllowedSkippedFrames = settings.m_minStaticTime * m_fps; // Maximum allowed skipped frames
            settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
        }
        else
        {
            settings.m_maximumAllowedSkippedFrames = 2 * m_fps; // Maximum allowed skipped frames
            settings.m_maxTraceLength = 4 * m_fps;              // Maximum trace length
        }

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsStatic())
            {
                DrawTrack(frame, 1, *track, true, true);
            }
            else
            {
                if (track->IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                    0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                    cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                        )
                {
                    DrawTrack(frame, 1, *track, true);
                }
            }
        }

        //m_detector->CalcMotionMap(frame);
    }

private:
    int m_minObjWidth;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetectorExample class
///
class FaceDetectorExample : public VideoExample
{
public:
    FaceDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        config_t config;
        config["cascadeFileName"] = "../data/haarcascade_frontalface_alt2.xml";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistJaccard;
        settings.m_kalmanType = tracking::KalmanUnscented;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;    // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;           // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = m_fps / 2;   // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The PedestrianDetectorExample class
///
class PedestrianDetectorExample : public VideoExample
{
public:
    PedestrianDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        tracking::Detectors detectorType = tracking::Detectors::Pedestrian_C4; // tracking::Detectors::Pedestrian_HOG;

        config_t config;
        config["detectorType"] = (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4";
        config["cascadeFileName1"] = "../data/combined.txt.model";
        config["cascadeFileName2"] = "../data/combined.txt.model_";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));


        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;    // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10;           // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = m_fps;   // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
			if (track->IsRobust(cvRound(m_fps / 2),          // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The SSDMobileNetExample class
///
class SSDMobileNetExample : public VideoExample
{
public:
    SSDMobileNetExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        config_t config;
        config["modelConfiguration"] = "../data/MobileNetSSD_deploy.prototxt";
        config["modelBinary"] = "../data/MobileNetSSD_deploy.caffemodel";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
        config["dnnTarget"] = "DNN_TARGET_OPENCL_FP16";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10;              // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = 2 * m_fps;  // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);

                std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief GrayProcessing
    /// \return
    ///
    bool GrayProcessing() const
    {
        return false;
    }
};

// ----------------------------------------------------------------------

///
/// \brief The YoloExample class
///
class YoloExample : public VideoExample
{
public:
    YoloExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        m_useLocalTracking = false;

        config_t config;
        //config["modelConfiguration"] = "../data/tiny-yolo.cfg";
        //config["modelBinary"] = "../data/tiny-yolo.weights";
        //config["modelConfiguration"] = "../data/yolov3-tiny.cfg";
        //config["modelBinary"] = "../data/yolov3-tiny.weights";
        //config["classNames"] = "../data/coco.names";
        config["modelConfiguration"] = "../../data/yolov3-urbantracker.cfg";
        config["modelBinary"] = "../../data/yolov3-urbantracker_13300.weights";
        config["classNames"] = "../../data/urbantracker.names";

        config["confidenceThreshold"] = "0.2";
        config["maxCropRatio"] = "3.0";
        config["dnnTarget"] = "DNN_TARGET_OPENCL_FP16";

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10;              // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = 2 * m_fps;  // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);

                std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief GrayProcessing
    /// \return
    ///
    bool GrayProcessing() const
    {
        return false;
    }
};

// ----------------------------------------------------------------------

///
/// \brief The CustomSSDMobileNetExample class
///
class CustomSSDMobileNetExample : public VideoExample
{
public:
	CustomSSDMobileNetExample(const cv::CommandLineParser& parser)
		:
		VideoExample(parser),
		mDeploy(parser.get<cv::String>("deploy")),
		mWeights(parser.get<cv::String>("weights")),
		mLabelMap(parser.get<cv::String>("label-map"))
	{
	}

protected:
	///
	/// \brief InitTracker
	/// \param grayFrame
	///
	bool InitTracker(cv::UMat frame)
	{
		config_t config;
		config["modelConfiguration"] = mDeploy.c_str();
		config["modelBinary"] = mWeights.c_str();
		config["labelMap"] = mLabelMap.c_str();
		config["meanValue"] = "104,117,123";
		config["confidenceThreshold"] = "0.5";
		config["maxCropRatio"] = "3.0";
		config["dnnTarget"] = "DNN_TARGET_OPENCL_FP16";
		m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_CustomNet, config, m_useLocalTracking, frame));
		if (!m_detector.get())
		{
			return false;
		}
		m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

		TrackerSettings settings;
		settings.m_useLocalTracking = m_useLocalTracking;
		settings.m_distType = tracking::DistRects;
		settings.m_kalmanType = tracking::KalmanLinear;
		settings.m_filterGoal = tracking::FilterRect;
		settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
		settings.m_matchType = tracking::MatchHungrian;
		settings.m_dt = 0.3f;                                // Delta time for Kalman filter
		settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
		settings.m_distThres = frame.rows / 10;              // Distance threshold between region and object on two frames
		settings.m_maximumAllowedSkippedFrames = 2 * m_fps;  // Maximum allowed skipped frames
		settings.m_maxTraceLength = 5 * m_fps;               // Maximum trace length

		m_tracker = std::make_unique<CTracker>(settings);

		return true;
	}

	///
	/// \brief DrawData
	/// \param frame
	///
	void DrawData(cv::Mat frame, int framesCounter, int currTime)
	{
		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
		}

		for (const auto& track : m_tracker->tracks)
		{
			if (track->IsRobust(5,                           // Minimal trajectory size
				0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, *track);

				//std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence;
				std::ostringstream label;
				label << track->m_lastRegion.m_type << ": " << std::setprecision(1) << track->m_lastRegion.m_confidence;
				int baseLine = 0;
				//cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				auto rect(track->GetLastRect());
				cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
				//cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
				cv::putText(frame, label.str(), cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		m_detector->CalcMotionMap(frame);
	}

	///
	/// \brief GrayProcessing
	/// \return
	///
	bool GrayProcessing() const
	{
		return false;
	}

private:
	cv::String mDeploy;
	cv::String mWeights;
	cv::String mLabelMap;
};