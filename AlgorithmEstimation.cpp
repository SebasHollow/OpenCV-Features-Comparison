#include "AlgorithmEstimation.hpp"

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;

    std::vector<float> distances (matches.size());
    for (size_t i = 0; i < matches.size(); i++)
        distances[i] = matches[i].distance;

    cv::Scalar mean, dev;
    meanStdDev (distances, mean, dev);

    meanDistance = static_cast<float> (mean.val[0]);
    stdDev       = static_cast<float> (dev.val[0]);

    return false;
}

float distance (const cv::Point2f& a, const cv::Point2f& b)
    {
    return sqrt((a - b).dot(a - b));
    }

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);

inline bool fileExists (const std::string& name)
    {
    struct stat buffer;
    return stat (name.c_str(), &buffer) == 0;
    }

bool performEstimation (const FeatureAlgorithm& alg, const ImageTransformation& transformation, const cv::Mat& sourceImage, const Keypoints& sourceKp,
                        const Descriptors& sourceDesc, std::vector<FrameMatchingStatistics>& stat)
    {
    std::vector<float> x = transformation.getX();
    stat.resize (x.size());

    const int count = x.size();

    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches;

    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();

    #pragma omp parallel for private(resKpReal, resDesc, matches) schedule(dynamic, 10)
    for (int i = 0; i < count; i++)
        {
        //std::cout << "Threads: " << omp_get_num_threads() << std::endl;
        float       arg = x[i];
        FrameMatchingStatistics& s = stat[i];

        const cv::String imgFilepath = R"(C:\TransformedImages\)" + transformation.name + std::to_string(i) + ".png";

        cv::Mat transformedImage;
        //if (fileExists (imgFilepath))
        //    transformedImage = imread (imgFilepath);
        //else
            //{
        transformation.transform (arg, sourceImage, transformedImage);
            //if (SAVE_TRANSFORMED_IMAGES)
            //    bool success = imwrite(imgFilepath, transformedImage);
            //}

        const cv::Mat expectedHomography = transformation.getHomography (arg, sourceImage);

        int64 start, end;
        size_t memoryAllocated;
        //cv::clearMemoryAllocated(); // Only works with custom compiled OpenCV version

        alg.extractFeatures (transformedImage, resKpReal, resDesc, start, end, memoryAllocated);

        // Initialize required fields
        s.memoryAllocated = memoryAllocated;
        s.isValid        = !resKpReal.empty();
        s.argumentValue  = arg;
        s.alg            = alg.name;
        s.trans          = transformation.name;
        if (!s.isValid)
            {
            std::cout << "Skipped for: " << alg.name << "\t" << transformation.name << "\t" << arg << std::endl;
            continue;
            }

        alg.matchFeatures (sourceDesc, resDesc, matches);

        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert (sourceKp, sourcePoints);
        cv::perspectiveTransform (sourcePoints, sourcePointsInFrame, expectedHomography);

        cv::Mat homography;

        int visibleFeatures = 0;
        int correctMatches  = 0;
        const int matchesCount = matches.size();

        for (const auto& point : sourcePoints)
            {
            if (point.x <= 0 ||
                point.y <= 0 ||
                point.x >= transformedImage.cols ||
                point.y >= transformedImage.rows)
                continue;

            visibleFeatures++;
            }

        for (auto& matche : matches)
            {
            const cv::Point2f expected = sourcePointsInFrame[matche.trainIdx];
            const cv::Point2f actual   = resKpReal[matche.queryIdx].pt;

            if (distance(expected, actual) < 3.0)
                correctMatches++;
            }

        s.totalKeypoints += resKpReal.size();
        s.consumedTimeMs += (end - start) * toMsMul;
        s.precision += correctMatches / static_cast<float>(matchesCount);
        s.recall += correctMatches / static_cast<float>(visibleFeatures);
        }

    return true;
    }

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography)
    {
    assert(!matches.empty());

    const int pointsCount = matches.size();
    std::vector<cv::Point2f> srcPoints, dstPoints;
    std::vector<float> distances;

    for (int i = 0; i < pointsCount; i++)
        {
        srcPoints.push_back(source[matches[i].trainIdx].pt);
        dstPoints.push_back(query[matches[i].queryIdx].pt);
        }

    perspectiveTransform(dstPoints, dstPoints, homography.inv());
    for (int i = 0; i < pointsCount; i++)
        {
        const cv::Point2f& src = srcPoints[i];
        const cv::Point2f& dst = dstPoints[i];

        cv::Point2f v = src - dst;
        distances.push_back(sqrtf(v.dot(v)));
        }


    cv::Scalar mean, dev;
    meanStdDev(distances, mean, dev);

    cv::Scalar result;
    result(0) = mean(0);
    result(1) = dev(0);
    result(2) = *std::max_element(distances.begin(), distances.end());
    result(3) = *std::min_element(distances.begin(), distances.end());
    return result;
    }
