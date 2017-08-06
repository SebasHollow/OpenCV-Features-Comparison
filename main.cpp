#include "ImageTransformation.hpp"
#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"
#include "AlgorithmEstimation.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cassert>

const bool USE_VERBOSE_TRANSFORMATIONS = false;
namespace fs = boost::filesystem;

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useBF = true;

    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("ORB",   cv::ORB::create(),   useBF));
    algorithms.push_back(FeatureAlgorithm("BRISK", cv::BRISK::create(), useBF));
    algorithms.push_back(FeatureAlgorithm("SURF",  cv::xfeatures2d::SURF::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("FREAK",  cv::xfeatures2d::FREAK::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("SIFT",  cv::xfeatures2d::SIFT::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("BRIEF",  cv::xfeatures2d::BriefDescriptorExtractor::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("LATCH",  cv::xfeatures2d::LATCH::create(),  useBF));

    transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(10)));
    cv::Ptr<ImageTransformation> rotationTransformation = cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 50, 10, cv::Point2f(0.5f, 0.5f)));
    transformations.push_back(rotationTransformation);
    cv::Ptr<ImageTransformation> scaleTransformation = cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.25f));
    transformations.push_back(scaleTransformation);
    transformations.push_back(cv::Ptr<ImageTransformation>(new CombinedTransform(rotationTransformation, scaleTransformation, CombinedTransform::ParamCombinationType::Full)));
    transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-125, +125, 25)));
    cv::Ptr<ImageTransformation> x = cv::Ptr<ImageTransformation>(new ImageXRotationTransformation(0, 45, 5, cv::Point2f(0.5f, 0.5f)));
    cv::Ptr<ImageTransformation> y = cv::Ptr<ImageTransformation>(new ImageYRotationTransformation(0, 45, 5, cv::Point2f(0.5f, 0.5f)));
    transformations.push_back(cv::Ptr<ImageTransformation>(new CombinedTransform(x, y, CombinedTransform::ParamCombinationType::Full)));

    if (argc < 2)
    {
        std::cout << "At least one input folder should be passed" << std::endl;
    }
    Keypoints sourceKp;
    Descriptors sourceDesc;
    cv::Mat sourceImage;
    cv::Ptr<cv::Feature2D> surf_detector = cv::xfeatures2d::SURF::create();
    std::string testImagePath;
    fs::path srcDir(argv[1]);
    fs::directory_iterator it(srcDir), eod;
    BOOST_FOREACH(fs::path const & testImagePath, std::make_pair(it, eod)) {
        std::string testImageName = testImagePath.filename().string();
        if (fs::is_regular_file(testImagePath) && testImageName[0] != '.') {
            std::cout << "Testing " << testImageName << std::endl;

            cv::Mat fullTestImage = cv::imread(testImagePath.string());

            cv::Mat testImage;

            if (fullTestImage.channels() == 3)
            {
                cv::cvtColor(fullTestImage, testImage, cv::COLOR_BGR2GRAY);
            }
            else if (fullTestImage.channels() == 4)
            {
                cv::cvtColor(fullTestImage, testImage, cv::COLOR_BGRA2GRAY);
            }
            else if (fullTestImage.channels() == 1)
            {
                testImage = fullTestImage;
            }

            surf_detector->detect(testImage, sourceKp);
            CollectedStatistics fullStat;

            if (testImage.empty())
            {
                std::cout << "Cannot read image from " << testImagePath << std::endl;
            }

            for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
            {
                cv::clearMemoryAllocated();
                const FeatureAlgorithm& alg   = algorithms[algIndex];
                Keypoints tempKp = sourceKp;
                sourceDesc = alg.getDescriptors(testImage, tempKp);
                std::cout << "Testing " << alg.name << "...";

                for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
                {
                    const ImageTransformation& trans = *transformations[transformIndex].get();
                    performEstimation(alg, trans, testImage.clone(), tempKp, sourceDesc, fullStat.getStatistics(alg.name, trans.name));
                }
                sourceDesc.release();
                std::cout << "done." << std::endl;
            }

            sourceKp.clear();

            fullStat.printAverage(std::cout, StatisticsElementRecall);
            fullStat.printAverage(std::cout, StatisticsElementPrecision);

            std::ofstream recallLog("Statistics/Recall_" + testImageName + ".txt");
            fullStat.printStatistics(recallLog, StatisticsElementRecall);

            std::ofstream precisionLog("Statistics/Precision_" + testImageName + ".txt");
            fullStat.printStatistics(precisionLog, StatisticsElementPrecision);

            std::ofstream memoryAllocatedLog("Statistics/MemoryAllocated_" + testImageName + ".txt");
            fullStat.printStatistics(memoryAllocatedLog, StatisticsElementMemoryAllocated);

            std::ofstream ConsumedTimeMsLog("Statistics/ConsumedTimeMs" + testImageName + ".txt");
            fullStat.printStatistics(ConsumedTimeMsLog, StatisticsElementConsumedTimeMs);

            std::ofstream memoryAllocatedPerDescriptorLog("Statistics/MemoryAllocatedPerDescriptor_" + testImageName + ".txt");
            fullStat.printStatistics(memoryAllocatedPerDescriptorLog, StatisticsElementMemoryAllocatedPerDescriptor);

            std::ofstream ConsumedTimeMsPerDescriptorLog("Statistics/ConsumedTimeMsPerDescriptor_" + testImageName + ".txt");
            fullStat.printStatistics(ConsumedTimeMsPerDescriptorLog, StatisticsElementConsumedTimeMsPerDescriptor);

            std::ofstream TotalKeypointsLog("Statistics/TotalKeypoints_" + testImageName + ".txt");
            fullStat.printStatistics(TotalKeypointsLog, StatisticsElementPointsCount);
        }
    }

    return 0;
}

