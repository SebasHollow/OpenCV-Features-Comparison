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

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useBF = true;

    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("ORB",   cv::ORB::create(),   useBF));
    //algorithms.push_back(FeatureAlgorithm("AKAZE", cv::Feature2D::create("AKAZE"), useBF));
    //algorithms.push_back(FeatureAlgorithm("KAZE",  cv::Feature2D::create("KAZE"),  useBF));
    algorithms.push_back(FeatureAlgorithm("BRISK", cv::BRISK::create(), useBF));
    algorithms.push_back(FeatureAlgorithm("SURF",  cv::xfeatures2d::SURF::create(),  useBF));
    //algorithms.push_back(FeatureAlgorithm("FREAK", cv::Ptr<cv::FeatureDetector>(new cv::SurfFeatureDetector(2000,4)), cv::Ptr<cv::DescriptorExtractor>(new cv::FREAK()), useBF));

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 1)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.01f)));
    }
    else
    {
        transformations.push_back(cv::Ptr<ImageTransformation>(new GaussianBlurTransform(9)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f, 0.5f))));
        transformations.push_back(cv::Ptr<ImageTransformation>(new ImageScalingTransformation(0.25f, 2.0f, 0.1f)));
        transformations.push_back(cv::Ptr<ImageTransformation>(new BrightnessImageTransform(-127, +127, 10)));
    }

    if (argc < 2)
    {
        std::cout << "At least one input image should be passed" << std::endl;
    }

    for (int imageIndex = 1; imageIndex < argc; imageIndex++)
    {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);

        CollectedStatistics fullStat;

        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
        {
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
    }

    return 0;
}

