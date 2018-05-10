#include "ImageTransformation.hpp"
#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"
#include "AlgorithmEstimation.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
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

const bool USE_VERBOSE_TRANSFORMATIONS = false;
namespace fs = boost::filesystem;

static std::vector<FeatureAlgorithm>              algorithms;
static std::vector<Ptr<ImageTransformation>> transformations;

void SURFTest();

void initializeAlgorithmsAndTransformations ()
    {
    bool useBF = true;

    // Initialize list of algorithm tuples
    algorithms.push_back (FeatureAlgorithm ("ORB", ORB::create(), useBF));
    algorithms.push_back (FeatureAlgorithm ("BRISK", BRISK::create(), useBF));
    algorithms.push_back (FeatureAlgorithm ("SURF", xfeatures2d::SURF::create(), useBF));
    //algorithms.push_back (FeatureAlgorithm ("FREAK",  xfeatures2d::FREAK::create(),  useBF));
    algorithms.push_back (FeatureAlgorithm ("SIFT", xfeatures2d::SIFT::create(), useBF));
    algorithms.push_back (FeatureAlgorithm ("BRIEF", xfeatures2d::BriefDescriptorExtractor::create(), useBF));
    algorithms.push_back (FeatureAlgorithm ("LATCH", xfeatures2d::LATCH::create(), useBF));

    transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform (15)));
    transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation (0, 90, 5, Point2f(0.5f, 0.5f))));
    transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation (0.5f, 2.0f, 0.25f)));

    const Ptr<ImageTransformation> rotationTransformation = cv::Ptr<ImageTransformation>(new ImageRotationTransformation (0, 45, 15, Point2f(0.5f, 0.5f)));
    const Ptr<ImageTransformation> scaleTransformation = cv::Ptr<ImageTransformation>(new ImageScalingTransformation (0.75f, 1.75f, 0.25f));
    transformations.push_back (cv::Ptr<ImageTransformation>(new CombinedTransform (scaleTransformation, rotationTransformation, CombinedTransform::ParamCombinationType::Full)));
    transformations.push_back (cv::Ptr<ImageTransformation>(new BrightnessImageTransform (-175, +175, 25)));

    Ptr<ImageTransformation> x = cv::Ptr<ImageTransformation>(new ImageXRotationTransformation (0, 40, 10, Point2f(0.5f, 0.5f)));
    Ptr<ImageTransformation> y = cv::Ptr<ImageTransformation>(new ImageYRotationTransformation (0, 40, 10, Point2f(0.5f, 0.5f)));
    transformations.push_back (cv::Ptr<ImageTransformation>(new CombinedTransform (x, y, CombinedTransform::ParamCombinationType::Full)));
    }

int main(int argc, const char* argv[])
    {
    //if (argc != 2)
    //    std::cout << "One input folder should be passed" << std::endl;

    //SURFTest();
    //return 0;

    std::string testPath;
    if (argc > 1)
        testPath = argv[1];
    else
        testPath = R"(C:\UniTools\Dataset\vot2016\glove\)";

    const fs::path srcDir (testPath);
    fs::directory_iterator it(srcDir), eod;
    Keypoints sourceKeypoints;
    Mat sourceImage;
    std::string testImagePath;
    CollectedStatistics fullStat;

    //std::vector<FeatureAlgorithm>              algorithms;
    //std::vector<cv::Ptr<ImageTransformation>> transformations;
    initializeAlgorithmsAndTransformations ();

    // Analysis happens here:
    Ptr<Feature2D> surf_detector = xfeatures2d::SURF::create(400);
    BOOST_FOREACH (fs::path const & testImagePath, std::make_pair(it, eod))
        {
        std::string testImageName = testImagePath.filename().string();
        if (is_regular_file(testImagePath) && testImageName[0] != '.')
            {
            std::cout << "Testing " << testImageName << std::endl;

            Mat fullTestImage = imread(testImagePath.string());
            Mat testImage;

            if (fullTestImage.channels() == 3)
                cvtColor(fullTestImage, testImage, COLOR_BGR2GRAY);
            else if (fullTestImage.channels() == 4)
                cvtColor(fullTestImage, testImage, COLOR_BGRA2GRAY);
            else if (fullTestImage.channels() == 1)
                testImage = fullTestImage;

            surf_detector->detect (testImage, sourceKeypoints);

            if (testImage.empty())
                std::cout << "Cannot read image from " << testImagePath << std::endl;

            // Try algorithms on the picture.
            for (const auto& alg : algorithms)
                {
                Keypoints tempKeypoints = sourceKeypoints;
                Descriptors sourceDescriptors = alg.getDescriptors (testImage, tempKeypoints);
                std::cout << "Testing " << alg.name << "...";

                // Apply transformations.
                for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
                    {
                    const ImageTransformation& trans = *transformations[transformIndex].get();
                    performEstimation (alg, trans, testImage.clone(), tempKeypoints, sourceDescriptors, fullStat.getStatistics (alg.name, trans.name));
                    }

                sourceDescriptors.release();
                std::cout << "done." << std::endl;
                }

            sourceKeypoints.clear();
            }

        std::ofstream recallLog("Recall_.txt");
        fullStat.printStatistics(recallLog, StatisticsElementRecall);

        std::ofstream precisionLog("Precision_.txt");
        fullStat.printStatistics(precisionLog, StatisticsElementPrecision);

        std::ofstream memoryAllocatedLog("MemoryAllocated_.txt");
        fullStat.printStatistics(memoryAllocatedLog, StatisticsElementMemoryAllocated);

        std::ofstream ConsumedTimeMsLog("ConsumedTimeMs.txt");
        fullStat.printStatistics(ConsumedTimeMsLog, StatisticsElementConsumedTimeMs);

        std::ofstream memoryAllocatedPerDescriptorLog("MemoryAllocatedPerDescriptor_.txt");
        fullStat.printStatistics(memoryAllocatedPerDescriptorLog, StatisticsElementMemoryAllocatedPerDescriptor);

        std::ofstream ConsumedTimeMsPerDescriptorLog("ConsumedTimeMsPerDescriptor_.txt");
        fullStat.printStatistics(ConsumedTimeMsPerDescriptorLog, StatisticsElementConsumedTimeMsPerDescriptor);

        std::ofstream TotalKeypointsLog("TotalKeypoints_.txt");
        fullStat.printStatistics(TotalKeypointsLog, StatisticsElementPointsCount);
        }

    fullStat.printAverage(std::cout, StatisticsElementRecall);
    fullStat.printAverage(std::cout, StatisticsElementPrecision);

    return 0;
    }

//void SURFTest ()
//    {
//
//    std::string testPath = R"(C:\UniTools\Dataset\vot2016\glove\)";
//
//    Mat img_1 = imread(R"(C:\UniTools\Dataset\vot2016\glove\00000001.jpg)", IMREAD_GRAYSCALE);
//    Mat img_2 = imread(R"(C:\UniTools\Dataset\vot2016\glove\00000002.jpg)", IMREAD_GRAYSCALE);
//    if (!img_1.data || !img_2.data)
//        {
//        std::cout << " --(!) Error reading images " << std::endl;
//        return;
//        }
//
//    int minHessian = 400;
//
//    Ptr<ORB> detector = ORB::create();
//
//    std::vector<KeyPoint> keypoints_1, keypoints_2;
//
//    detector->detect(img_1, keypoints_1);
//    detector->detect(img_2, keypoints_2);
//
//    //-- Draw keypoints
//    Mat img_keypoints_1; Mat img_keypoints_2;
//
//    drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//    drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//
//    //-- Show detected (drawn) keypoints
//    imshow("Keypoints 1", img_keypoints_1);
//    imshow("Keypoints 2", img_keypoints_2);
//    }
