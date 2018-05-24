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

void DrawMatches();
void TestImage (const Mat& testImage, CollectedStatistics& statistics);
void FindAndSaveKeypoints (const ImageData data);

static std::vector<FeatureAlgorithm> algorithms;
static std::vector<Ptr<ImageTransformation>> transformations;
static Ptr<Feature2D> surf_detector = xfeatures2d::SURF::create();

const std::string _defaultTestDir = R"(C:\Dataset\)";

const std::vector<float> scalingArgs = { 0.25, 0.5, 0.75, 2, 3, 4 };
const std::vector<float> perspectiveRotationArgs = { 15, 30, 45, 60, 75 };

void initializeTransformations()
    {
    //transformations.push_back (cv::Ptr<ImageTransformation> (new GaussianBlurTransform (5, 30, 5)));
    //transformations.push_back (cv::Ptr<ImageTransformation> (new ImageRotationTransformation (15, 180, 15, Point2f (0.5f, 0.5f))));
    //transformations.push_back (cv::Ptr<ImageTransformation> (new ImageScalingTransformation (scalingArgs)));
    //transformations.push_back (cv::Ptr<ImageTransformation> (new BrightnessTransform (-125, +125, 25)));

    transformations.push_back (cv::Ptr<ImageTransformation> (new PerspectiveTransform (perspectiveRotationArgs, "Perspective rotation")));

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
            std::cout << "Cannot read imageOriginal from " << testImagePath << std::endl;
            continue;
            }

        const auto originalImage = imread (testImagePath.string());
        auto testImage = ConvertImage (originalImage);
        if (testImage.empty())
            {
            std::cout << "Cannot read imageOriginal from " << testImagePath << std::endl;
            continue;
            }

        ImageData srcData;
        srcData.image = testImageName;
        srcData.imageOriginal = originalImage;
        srcData.imageGrey = testImage;

        if (SAVE_IMAGES)
            FindAndSaveKeypoints (srcData);

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

    if (SAVE_IMAGES)
        DrawMatches ();

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

void FindAndSaveKeypoints (const ImageData data)
    {
    Keypoints sourceKeypoints;
    surf_detector->detect (data.imageOriginal, sourceKeypoints);
    Mat keypointPicture;

    drawKeypoints (data.imageOriginal, sourceKeypoints, keypointPicture);
    const String imgFilepath = R"(C:\TransformedImages\)" + data.image + "(Keypoints).png";
    imwrite (imgFilepath, keypointPicture);

    drawKeypoints (data.imageOriginal, sourceKeypoints, keypointPicture, -1, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    const String richImgFilepath = R"(C:\TransformedImages\)" + data.image + "(Rich keypoints).png";
    imwrite (richImgFilepath, keypointPicture);
    }

void DrawMatches ()
    {
    const auto pic1 = imread (R"(C:\avatars\masked.png)");
    //const auto pic2 = imread (R"(C:\avatars\masked.png)");
    const auto pic2 = imread (R"(C:\avatars\masked (circle).png)");

    Matches matches;
    Keypoints keypoints1, keypoints2;
    Descriptors descriptors1, descriptors2;
    surf_detector->detect (pic1, keypoints1);
    surf_detector->detect (pic2, keypoints2);

    for (const auto& alg : algorithms)
        {
        int64 start, end;
        alg.extractFeatures (pic1, keypoints1, descriptors1, start, end);
        alg.extractFeatures (pic2, keypoints2, descriptors2, start, end);
        alg.matchFeatures (descriptors1, descriptors2, matches);

        Mat outPic;
        drawMatches (pic2, keypoints2, pic1, keypoints1, matches, outPic);
        imwrite (R"(C:\TransformedImages\Match example with )" + alg.name + " descriptor.png", outPic);
        }
    }
