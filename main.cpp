#include "ImageTransformation.hpp"
#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"
#include "AlgorithmEstimation.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <numeric>
#include <fstream>
#include <cassert>

#include <iostream>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

void PrintLogs (const CollectedStatistics& stats);
Mat ConvertImage (const Mat& fullTestImage);
void TestImage (const Mat& testImage, CollectedStatistics& statistics);
void CreateLogsDir();

static std::vector<FeatureAlgorithm> algorithms;
static std::vector<Ptr<ImageTransformation>> transformations;
static Ptr<Feature2D> surf_detector = xfeatures2d::SURF::create();

const std::string _defaultTestDir = R"(C:\Dataset\)";
const std::string _logsDir = R"(logs\)";

const std::vector<float> scalingArgs = { 0.25, 0.5, 0.75, 2, 3, 4};

void initializeTransformations()
    {
    transformations.push_back (cv::Ptr<ImageTransformation> (new GaussianBlurTransform (5, 30, 5)));
    transformations.push_back (cv::Ptr<ImageTransformation> (new ImageRotationTransformation (15, 180, 15, Point2f (0.5f, 0.5f))));
    transformations.push_back (cv::Ptr<ImageTransformation> (new ImageScalingTransformation (scalingArgs)));
    transformations.push_back (cv::Ptr<ImageTransformation> (new BrightnessTransform (-125, +125, 25)));

    //transformations.push_back (cv::Ptr<ImageTransformation> (new PerspectiveTransform (5, "Z Perspective")));

    //const auto x = cv::Ptr<ImageTransformation>(new ImageXRotationTransformation (10, 60, 50, Point2f (0.5f, 0.5f)));
    //const auto y = cv::Ptr<ImageTransformation>(new ImageYRotationTransformation (10, 20, 50, Point2f (0.5f, 0.5f)));
    //transformations.push_back (cv::Ptr<ImageTransformation> (new CombinedTransform (x, y, CombinedTransform::ParamCombinationType::Full)));

    //const auto rotationTransformation = cv::Ptr<ImageTransformation> (new ImageRotationTransformation (0, 45, 15, Point2f (0.5f, 0.5f)));
    //transformations.push_back (rotationTransformation);
    //const auto scaleTransformation = cv::Ptr<ImageTransformation> (new ImageScalingTransformation (0.75f, 1.75f, 0.25f));
    //transformations.push_back(cv::Ptr<ImageTransformation> (new CombinedTransform (scaleTransformation, rotationTransformation, CombinedTransform::ParamCombinationType::Full)));
    }

void initializeAlgorithms()
    {
    bool useBF = true;

    // Initialize list of algorithm tuples
    algorithms.emplace_back ("SIFT", xfeatures2d::SIFT::create(), useBF);
    algorithms.emplace_back ("SURF", xfeatures2d::SURF::create(), useBF);
    algorithms.emplace_back ("ORB", ORB::create(), useBF);
    algorithms.emplace_back ("BRISK", BRISK::create(), useBF);
    algorithms.emplace_back ("BRIEF", xfeatures2d::BriefDescriptorExtractor::create(), useBF);
    algorithms.emplace_back ("LATCH", xfeatures2d::LATCH::create(), useBF);
    }

int main (int argc, const char* argv[])
    {
    initializeTransformations();
    initializeAlgorithms();

    std::string testPath;
    if (argc > 1)
        testPath = argv[1];
    else
        testPath = _defaultTestDir;

    CreateLogsDir();

    const fs::path srcDir (testPath);
    fs::directory_iterator it (srcDir), eod;
    CollectedStatistics fullStat;

    int imageCount = 0;
    // Analysis happens here:
    BOOST_FOREACH (fs::path const & testImagePath, std::make_pair(it, eod))
        {
        auto testImageName = testImagePath.filename().string();
        if (!is_regular_file (testImagePath) || testImageName[0] == '.')
            {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
            continue;
            }

        auto testImage = ConvertImage (imread (testImagePath.string()));
        if (testImage.empty())
            {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
            continue;
            }

        std::cout << "Testing picture " << ++imageCount << ": " << testImageName << std::endl;
        TestImage (testImage, fullStat);

        PrintLogs (fullStat);
        }

    return 0;
    }

Mat ConvertImage (const Mat& fullTestImage)
    {
    Mat testImage;

    switch (fullTestImage.channels())
        {
    case 3:
        cvtColor (fullTestImage, testImage, COLOR_BGR2GRAY);
        return testImage;
    case 4:
        cvtColor (fullTestImage, testImage, COLOR_BGRA2GRAY);
        return testImage;
    case 1:
        return fullTestImage;
    default:
        return testImage;
        }
    }

void TestImage (const Mat& testImage, CollectedStatistics& statistics)
    {
    Keypoints sourceKeypoints;
    surf_detector->detect (testImage, sourceKeypoints);

    for (const auto& alg : algorithms)
        {
        auto tempKeypoints = sourceKeypoints;
        auto sourceDescriptors = alg.getDescriptors (testImage, tempKeypoints);
        std::cout << "Testing " << alg.name << "...";

        // Apply transformations.
        for (auto& transformation : transformations)
            {
            const ImageTransformation& trans = *transformation;
            performEstimation (alg, trans, testImage.clone(), tempKeypoints, sourceDescriptors, statistics.getStatistics (alg.name, trans.name));
            }

        sourceDescriptors.release();
        std::cout << "done." << std::endl;
        }

    sourceKeypoints.clear();
    std::cout << std::endl;
    }

void PrintLogs (const CollectedStatistics& stats)
    {
    std::ofstream recallLog (_logsDir + "Recall_.txt");
    stats.printStatistics (recallLog, StatisticsElementRecall);

    std::ofstream precisionLog (_logsDir + "Precision_.txt");
    stats.printStatistics (precisionLog, StatisticsElementPrecision);

    std::ofstream memoryAllocatedLog (_logsDir + "MemoryAllocated_.txt");
    stats.printStatistics (memoryAllocatedLog, StatisticsElementMemoryAllocated);

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

void CreateLogsDir ()
    {
    const char* path = _logsDir.c_str();
    const boost::filesystem::path dir (path);
    if (create_directory (dir))
        std::cout << "Directory Created: " << _logsDir << std::endl;
    }
