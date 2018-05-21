#include "Util.hpp"
#include <opencv2/core/mat.hpp>
#include <boost/filesystem/operations.hpp>

const std::string _logsDir = R"(logs\)";

cv::Mat ConvertImage (const cv::Mat& fullTestImage)
    {
    cv::Mat testImage;

    switch (fullTestImage.channels())
        {
        case 3:
            cvtColor(fullTestImage, testImage, cv::COLOR_BGR2GRAY);
            return testImage;
        case 4:
            cvtColor(fullTestImage, testImage, cv::COLOR_BGRA2GRAY);
            return testImage;
        case 1:
            return fullTestImage;
        default:
            return testImage;
        }
    }

void PrintLogs (const CollectedStatistics& stats)
    {
    std::ofstream recallLog (_logsDir + "Recall_.txt");
    stats.printStatistics (recallLog, StatisticsElementRecall);

    std::ofstream precisionLog (_logsDir + "Precision_.txt");
    stats.printStatistics (precisionLog, StatisticsElementPrecision);

    std::ofstream ConsumedTimeMsLog (_logsDir + "ConsumedTimeMs.txt");
    stats.printStatistics (ConsumedTimeMsLog, StatisticsElementConsumedTimeMs);

    std::ofstream memoryAllocatedPerDescriptorLog (_logsDir + "MemoryAllocatedPerDescriptor_.txt");
    stats.printStatistics (memoryAllocatedPerDescriptorLog, StatisticsElementMemoryAllocatedPerDescriptor);

    std::ofstream ConsumedTimeMsPerDescriptorLog (_logsDir + "ConsumedTimeMsPerDescriptor_.txt");
    stats.printStatistics (ConsumedTimeMsPerDescriptorLog, StatisticsElementConsumedTimeMsPerDescriptor);

    std::ofstream TotalKeypointsLog (_logsDir + "TotalKeypoints_.txt");
    stats.printStatistics (TotalKeypointsLog, StatisticsElementPointsCount);

    std::ofstream statisticsElementRecall (_logsDir + "Average_StatisticsElementRecall_.txt");
    stats.printAverage (statisticsElementRecall, StatisticsElementRecall);

    std::ofstream statisticsElementPrecision (_logsDir + "Average_StatisticsElementPrecision_.txt");
    stats.printAverage (statisticsElementPrecision, StatisticsElementPrecision);

    std::ofstream performanceStatistics (_logsDir + "performanceStatistics_.txt");
    stats.printPerformanceStatistics (performanceStatistics);
    }

void CreateLogsDir()
    {
    const char* path = _logsDir.c_str();
    const boost::filesystem::path dir (path);
    if (create_directory (dir))
        std::cout << "Directory Created: " << _logsDir << std::endl;
    }


inline bool fileExists(const std::string& name)
    {
    struct stat buffer;
    return stat(name.c_str(), &buffer) == 0;
    }

