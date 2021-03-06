#include "AlgorithmEstimation.hpp"
#include "Util.hpp"

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

cv::Scalar computeReprojectionError (const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);

bool performEstimation (const FeatureAlgorithm& alg, const ImageTransformation& transformation, const cv::Mat& srcImage, const Keypoints& srcKeypoints,
                        const Descriptors& srcDescriptors, std::vector<FrameMatchingStatistics>& stat)
    {
    std::vector<float> x = transformation.getX();
    stat.resize (x.size());

    const int count = x.size();

    Keypoints   resKeypoints;
    Descriptors resDescriptors;

    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();

    #pragma omp parallel for private (resKpReal, resDesc, matches) schedule(dynamic, 10)
    for (int i = 0; i < count; i++)
        {
        //std::cout << "Threads: " << omp_get_num_threads() << std::endl;
        const float arg = x[i];

        cv::Mat transformedImage;
        transformation.transform (arg, srcImage, transformedImage);
        const cv::Mat expectedHomography = transformation.getHomography (arg, srcImage);

        if (SAVE_IMAGES)
            imwrite (R"(C:\TransformedImages\)" + transformation.name + " (" + std::to_string(arg) + ").png", transformedImage);

        int64 start, end;
        const bool success = alg.extractFeatures (transformedImage, resKeypoints, resDescriptors, start, end);
        if (!success || resKeypoints.empty())
            {
            std::cout << "Skipped for: " << alg.name << "\t" << transformation.name << "\t" << arg << std::endl;
            continue;
            }

        Matches matches;
        alg.matchFeatures (srcDescriptors, resDescriptors, matches);

        // Calculate source points and source points in expected homography's frame.
        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert (srcKeypoints, sourcePoints);
        perspectiveTransform (sourcePoints, sourcePointsInFrame, expectedHomography);

        // Count visible features and correct matches.
        const int visibleFeatures = CountVisibleFeatures (sourcePoints, transformedImage.cols, transformedImage.rows);
        const int correctMatches = CountCorrectMatches (matches, sourcePointsInFrame, resKeypoints);

        FrameMatchingStatistics& s = stat[i];
        // Initialize required fields
        s.isValid = !resKeypoints.empty();
        s.argumentValue = arg;
        s.alg = alg.name;
        s.trans = transformation.name;
        // Fill in the remaining statistics.
        s.totalKeypoints += resKeypoints.size();
        s.consumedTimeMs += (end - start) * toMsMul;
        s.precision += correctMatches / static_cast<float>(matches.size());
        s.recall += correctMatches / static_cast<float>(visibleFeatures);
        }

    return true;
    }

bool performEstimation (const FeatureAlgorithm& alg, const ImageTransformation& transformation, ImageData src, std::vector<FrameMatchingStatistics>& stat)
    {
    std::vector<float> x = transformation.getX();
    stat.resize (x.size());
    const int count = x.size();

    ImageData res;

    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();

    #pragma omp parallel for private (resKpReal, resDesc, matches) schedule(dynamic, 10)
    for (int i = 0; i < count; i++)
        {
        //std::cout << "Threads: " << omp_get_num_threads() << std::endl;
        const float arg = x[i];

        cv::Mat transformedImage;
        transformation.transform (arg, src.imageGrey, transformedImage);
        const cv::Mat expectedHomography = transformation.getHomography (arg, src.imageGrey);

        if (SAVE_IMAGES)
            imwrite (R"(C:\TransformedImages\)" + src.image + " with " + transformation.name + " (" + std::to_string(arg) + ").png", transformedImage);

        int64 start, end;
        const bool success = alg.extractFeatures (transformedImage, res.keypoints, res.descriptors, start, end);
        if (!success || res.keypoints.empty())
            {
            std::cout << "Skipped for: " << alg.name << "\t" << transformation.name << "\t" << arg << std::endl;
            continue;
            }

        Matches matches;
        alg.matchFeatures (src.descriptors, res.descriptors, matches);

        if (SAVE_IMAGES)
            {
            cv::Mat outPic;
            drawMatches (src.imageOriginal, src.keypoints, transformedImage, res.keypoints,  matches, outPic);
            imwrite (R"(C:\TransformedImages\)" + src.image + " matches with " + transformation.name + " (" + std::to_string(arg) + ").png", transformedImage);

            continue;
            }

        // Calculate source points and source points in expected homography's frame.
        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert (src.keypoints, sourcePoints);
        perspectiveTransform (sourcePoints, sourcePointsInFrame, expectedHomography);

        // Count visible features and correct matches.
        const int visibleFeatures = CountVisibleFeatures (sourcePoints, transformedImage.cols, transformedImage.rows);
        const int correctMatches = CountCorrectMatches (matches, sourcePointsInFrame, res.keypoints);

        FrameMatchingStatistics& s = stat[i];
        // Initialize required fields
        s.isValid = !res.keypoints.empty();
        s.argumentValue = arg;
        s.alg = alg.name;
        s.trans = transformation.name;
        // Fill in the remaining statistics.
        s.totalKeypoints += res.keypoints.size();
        s.consumedTimeMs += (end - start) * toMsMul;
        s.precision += correctMatches / static_cast<float>(matches.size());
        s.recall += correctMatches / static_cast<float>(visibleFeatures);
        }

    return true;
    }


int CountVisibleFeatures (std::vector<cv::Point2f>& sourcePoints, int imageCols, int imageRows)
    {
    int visibleFeatures = 0;

    for (const auto& point : sourcePoints)
        {
        if (point.x <= 0 ||
            point.y <= 0 ||
            point.x >= imageCols ||
            point.y >= imageRows)
            continue;

        visibleFeatures++;
        }

    return visibleFeatures;
    }

int CountCorrectMatches (Matches& matches, std::vector<cv::Point2f>& sourcePointsInFrame, Keypoints& resKpReal)
    {
    int correctMatches = 0;
    const int matchesCount = matches.size();

    for (auto& match : matches)
        {
        const cv::Point2f expected = sourcePointsInFrame[match.trainIdx];
        const cv::Point2f actual = resKpReal[match.queryIdx].pt;

        if (distance (expected, actual) < 3.0)
            correctMatches++;
        }

    return correctMatches;
    }

cv::Scalar computeReprojectionError (const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography)
    {
    assert (!matches.empty());

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
