#ifndef UTIL_HPP
#define UTIL_HPP
#include <opencv2/core/mat.hpp>
#include "CollectedStatistics.hpp"

cv::Mat ConvertImage(const cv::Mat& fullTestImage);

void PrintLogs (const CollectedStatistics& stats);

void CreateLogsDir();

#endif
