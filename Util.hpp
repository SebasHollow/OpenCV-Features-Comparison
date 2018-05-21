#ifndef UTIL_HPP
#define UTIL_HPP
#include <opencv2/core/mat.hpp>
#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"

cv::Mat ConvertImage(const cv::Mat& fullTestImage);

void PrintLogs (const CollectedStatistics& stats);

void CreateLogsDir();


struct ImageData {
    cv::Mat imageOriginal;
    std::string image;
    cv::Mat imageGrey;
    Keypoints keypoints;
    Descriptors descriptors;
};


#endif
