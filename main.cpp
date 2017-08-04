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

const bool USE_VERBOSE_TRANSFORMATIONS = false;
namespace fs = boost::filesystem;

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useBF = true;
    cv::fastMalloc(2);
    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("ORB",   cv::ORB::create(),   useBF));
    algorithms.push_back(FeatureAlgorithm("BRISK", cv::BRISK::create(), useBF));
    algorithms.push_back(FeatureAlgorithm("SURF",  cv::xfeatures2d::SURF::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("FREAK",  cv::xfeatures2d::FREAK::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("SIFT",  cv::xfeatures2d::SIFT::create(),  useBF));
    algorithms.push_back(FeatureAlgorithm("BRIEF",  cv::xfeatures2d::BriefDescriptorExtractor::create(),  useBF));

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 1)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.01f)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new PerspectiveTransform(10)));
    }
    else
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.1f)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 10)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new PerspectiveTransform(10)));
    }

    if (argc < 2)
    {
        std::cout << "At least one input image should be passed" << std::endl;
    }

    std::string testImagePath;
    fs::path srcDir(argv[1]);
    fs::directory_iterator it(srcDir), eod;
    BOOST_FOREACH(fs::path const & testImagePath, std::make_pair(it, eod)) {
        std::string testImageName = testImagePath.filename().string();
        if (fs::is_regular_file(testImagePath) && testImageName[0] != '.') {
            std::cout << "Testing " << testImageName << std::endl;
            
            cv::Mat testImage = cv::imread(testImagePath.string());

            CollectedStatistics fullStat;

            if (testImage.empty())
            {
                std::cout << "Cannot read image from " << testImagePath << std::endl;
            }

            for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
            {
                cv::clearMemoryAllocated();
                const FeatureAlgorithm& alg   = algorithms[algIndex];

                std::cout << "Testing " << alg.name << "...";

                for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
                {
                    const ImageTransformation& trans = *transformations[transformIndex].get();

                    performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));
                }

                std::cout << "done." << std::endl;
            }

            fullStat.printAverage(std::cout, StatisticsElementRecall);
            fullStat.printAverage(std::cout, StatisticsElementPrecision);

            std::ofstream recallLog("Recall.txt");
            fullStat.printStatistics(recallLog, StatisticsElementRecall);

            std::ofstream precisionLog("Precision.txt");
            fullStat.printStatistics(precisionLog, StatisticsElementPrecision);

            std::ofstream performanceLog("Performance.txt");
            fullStat.printPerformanceStatistics(performanceLog);

            std::ofstream memoryAllocatedLog("MemoryAllocated.txt");
            fullStat.printStatistics(memoryAllocatedLog, StatisticsElementMemoryAllocated);
        }
    }

    return 0;
}

