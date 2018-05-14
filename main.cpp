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
void initializeAlgorithmsAndTransformations ();

static std::vector<FeatureAlgorithm> algorithms;
static std::vector<Ptr<ImageTransformation>> transformations;
static Ptr<Feature2D> surf_detector = xfeatures2d::SURF::create();
const bool USE_VERBOSE_TRANSFORMATIONS = true;

const std::string _defaultTestDir = R"(C:\Dropbox\Bakalauras\.workdir\Datasets\Resolution\dataset)";
const std::string _logsDir = R"(logs\)";

void CreateLogsDir ();

int main (int argc, const char* argv[])
    {
    initializeAlgorithmsAndTransformations();

    std::string testPath;
    if (argc > 1)
        testPath = argv[1];
    else
        testPath = _defaultTestDir;
    
    CreateLogsDir();

    const fs::path srcDir (testPath);
    fs::directory_iterator it (srcDir), eod;
    CollectedStatistics fullStat;

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

        std::cout << "Testing " << testImageName << std::endl;
        TestImage (testImage, fullStat);

        PrintLogs (fullStat);
        }

    fullStat.printAverage (std::cout, StatisticsElementRecall);
    fullStat.printAverage (std::cout, StatisticsElementPrecision);

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
            performEstimation (alg, trans, testImage.clone(), tempKeypoints, sourceDescriptors,
                               statistics.getStatistics (alg.name, trans.name));
            }

        sourceDescriptors.release();
        std::cout << "done." << std::endl;
        }

    sourceKeypoints.clear();
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

    std::ofstream TotalKeypointsLog ("TotalKeypoints_.txt");
    stats.printStatistics (TotalKeypointsLog, StatisticsElementPointsCount);
    }

void initializeAlgorithmsAndTransformations ()
    {
    bool useBF = true;

    // Initialize list of algorithm tuples
    algorithms.emplace_back ("ORB", ORB::create(), useBF);
    algorithms.emplace_back ("BRISK", BRISK::create(), useBF);
    algorithms.emplace_back ("SURF", xfeatures2d::SURF::create(), useBF);
    //algorithms.push_back  (FeatureAlgorithm ("FREAK",  xfeatures2d::FREAK::create(),  useBF));
    algorithms.emplace_back ("SIFT", xfeatures2d::SIFT::create(), useBF);
    algorithms.emplace_back ("BRIEF", xfeatures2d::BriefDescriptorExtractor::create(), useBF);
    algorithms.emplace_back ("LATCH", xfeatures2d::LATCH::create(), useBF);

    transformations.push_back (cv::Ptr<ImageTransformation> (new GaussianBlurTransform (15)));
    transformations.push_back (
        cv::Ptr<ImageTransformation> (new ImageRotationTransformation (0, 90, 5, Point2f (0.5f, 0.5f))));
    transformations.push_back (cv::Ptr<ImageTransformation> (new ImageScalingTransformation (0.5f, 2.0f, 0.25f)));

    const Ptr<ImageTransformation> rotationTransformation = cv::Ptr<ImageTransformation> (
        new ImageRotationTransformation (0, 45, 15, Point2f (0.5f, 0.5f)));
    const Ptr<ImageTransformation> scaleTransformation = cv::Ptr<ImageTransformation> (
        new ImageScalingTransformation (0.75f, 1.75f, 0.25f));
    transformations.push_back (cv::Ptr<ImageTransformation> (
        new CombinedTransform (scaleTransformation, rotationTransformation,
                               CombinedTransform::ParamCombinationType::Full)));
    transformations.push_back (cv::Ptr<ImageTransformation> (new BrightnessImageTransform (-175, +175, 25)));

    Ptr<ImageTransformation> x = cv::Ptr<ImageTransformation> (
        new ImageXRotationTransformation (0, 40, 10, Point2f (0.5f, 0.5f)));
    Ptr<ImageTransformation> y = cv::Ptr<ImageTransformation> (
        new ImageYRotationTransformation (0, 40, 10, Point2f (0.5f, 0.5f)));
    transformations.push_back (
        cv::Ptr<ImageTransformation> (new CombinedTransform (x, y, CombinedTransform::ParamCombinationType::Full)));
    }

void CreateLogsDir()
    {
    const char* path = _logsDir.c_str();
    const boost::filesystem::path dir (path);
    if (create_directory(dir))
        std::cerr << "Directory Created: " << _logsDir << std::endl;
    }
