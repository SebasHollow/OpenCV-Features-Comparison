#include "ImageTransformation.hpp"
#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"
#include "AlgorithmEstimation.hpp"
#include "Util.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <numeric>
#include <cassert>

#include <iostream>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

void TestImage (const Mat& testImage, CollectedStatistics& statistics);


static std::vector<FeatureAlgorithm> algorithms;
static std::vector<Ptr<ImageTransformation>> transformations;
static Ptr<Feature2D> surf_detector = xfeatures2d::SURF::create();

const std::string _defaultTestDir = R"(C:\Dataset\)";

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
    freopen ("log.txt", "w", stdout);

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

void TestImage (const Mat& testImage, CollectedStatistics& statistics)
    {
    const auto startTime = std::chrono::system_clock::now();

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
        std::cout << std::endl;
        }

    sourceKeypoints.clear();

    // Duration logging
    const auto endTime = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s";
    std::cout << std::endl;
    }
