#include "AlgorithmEstimation.hpp"
#include <fstream>
#include <iterator>
#include <cstdint>
#include <omp.h>

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;

    std::vector<float> distances(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
        distances[i] = matches[i].distance;

    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);

    meanDistance = static_cast<float>(mean.val[0]);
    stdDev       = static_cast<float>(dev.val[0]);

    return false;
}

float distance(const cv::Point2f a, const cv::Point2f b)
{
    return sqrt((a - b).dot(a - b));
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography);


bool performEstimation
(
    const FeatureAlgorithm& alg,
    const ImageTransformation& transformation,
    const cv::Mat& sourceImage,
    const Keypoints& sourceKp,
    const Descriptors& sourceDesc,
    std::vector<FrameMatchingStatistics>& stat
)
{

// Keypoints   sourceKp;
//     Descriptors sourceDesc;

//     cv::Mat gray;

//     if (sourceImage.channels() == 3)
//     {
//         cv::cvtColor(sourceImage, gray, cv::COLOR_BGR2GRAY);
//     }
//     else if (sourceImage.channels() == 4)
//     {
//         cv::cvtColor(sourceImage, gray, cv::COLOR_BGRA2GRAY);
//     }
//     else if(sourceImage.channels() == 1)
//     {
//         gray = sourceImage;
//     }

//     if (!alg.extractFeatures(gray, sourceKp, sourceDesc))
//         return false;

    //

    std::vector<float> x = transformation.getX();
    stat.resize(x.size() * (T_MAX / T_STEP ));
    int statsCounter = 0;
    const int count = x.size();

    Keypoints   resKpReal;
    Descriptors resDesc;
    Matches     matches;

    // To convert ticks to milliseconds
    const double toMsMul = 1000. / cv::getTickFrequency();

    #pragma omp parallel for private(resKpReal, resDesc, matches) schedule(dynamic, 10)
    for (int i = 0; i < count; i++)
    {
        float       arg = x[i];

        cv::Mat     transformedImage;
        transformation.transform(arg, sourceImage, transformedImage);

        if (0)
        {
            //std::ostringstream image_name;
            cv::imwrite("Destination/" + transformation.name + std::to_string(i) + ".png", transformedImage);
            // image_name << "image_dump_" << transformation.name << "_" << i << ".bin";
            // std::ofstream dump(image_name.str().c_str(), std::ios::binary);
            // std::copy(transformedImage.datastart, transformedImage.dataend, std::ostream_iterator<uint8_t>(dump));
        }

        cv::Mat expectedHomography = transformation.getHomography(arg, sourceImage);

        int64 start, end;
        size_t memoryAllocated;

        alg.extractFeatures(transformedImage, resKpReal, resDesc, start, end, memoryAllocated);

        if (resKpReal.size() <= 0)
            continue;

        try
        {
            alg.matchFeatures(sourceDesc, resDesc, matches);
        }
        catch ( cv::Exception& e )
        {
            const char* err_msg = e.what();
            std::cout << "Exception for " << transformation.name << i << ": " << err_msg << std::endl;
            continue;
        }

        std::vector<cv::Point2f> sourcePoints, sourcePointsInFrame;
        cv::KeyPoint::convert(sourceKp, sourcePoints);
        cv::perspectiveTransform(sourcePoints, sourcePointsInFrame, expectedHomography);

        cv::Mat homography;

        //so, we have :
        //N - number of keypoints in the first image that are also visible
        //    (after transformation) on the second image

        //    N1 - number of keypoints in the first image that have been matched.

        //    n - number of the correct matches found by the matcher

        //    n / N1 - precision
        //    n / N - recall(? )

        int visibleFeatures = 0;
        int correctMatches  = 0;
        int matchesCount    = matches.size();

        for (int i = 0; i < sourcePoints.size(); i++)
        {
            if (sourcePointsInFrame[i].x > 0 &&
                    sourcePointsInFrame[i].y > 0 &&
                    sourcePointsInFrame[i].x < transformedImage.cols &&
                    sourcePointsInFrame[i].y < transformedImage.rows)
            {
                visibleFeatures++;
            }
        }
        //std::cout << "NEW PERM" << std::endl;
        for (float t = T_STEP; t <= T_MAX; t += T_STEP) {
            for (int i = 0; i < matches.size(); i++)
            {
                cv::Point2f expected = sourcePointsInFrame[matches[i].trainIdx];
                cv::Point2f actual   = resKpReal[matches[i].queryIdx].pt;

                if (distance(expected, actual) < t)
                {
                    correctMatches++;
                }
            }
            FrameMatchingStatistics& s = stat[statsCounter];
            s.isValid        = resKpReal.size() > 0;
            s.argumentValue  = arg;
            s.threshold      = t;
            s.alg            = alg.name;
            s.trans          = transformation.name;
            s.memoryAllocated = memoryAllocated;
            s.totalKeypoints = resKpReal.size();
            s.consumedTimeMs = (end - start) * toMsMul;
            s.precision = correctMatches / (float) matchesCount;
            s.recall = correctMatches / (float) visibleFeatures; // correctMatches +
            //std::cout << statsCounter << std::endl;
            statsCounter++;
            correctMatches  = 0;
        }
        //std::cout << "matchesCount: " << matchesCount << ", visibleFeatures: " << visibleFeatures << std::endl;

        //bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, correctMatches, homography);

        // Some simple stat:
        //s.isValid        = homographyFound;

        // Compute matching statistics
        //if (homographyFound)
        //{
        //    cv::Mat r = expectedHomography * homography.inv();
        //    float error = cv::norm(cv::Mat::eye(3,3, CV_64FC1) - r, cv::NORM_INF);

        //    computeMatchesDistanceStatistics(correctMatches, s.meanDistance, s.stdDevDistance);
        //    s.reprojectionError = computeReprojectionError(sourceKp, resKpReal, correctMatches, homography);
        //    s.homographyError = std::min(error, 1.0f);

        //    if (0 && error >= 1)
        //    {
        //        std::cout << "H expected:" << expectedHomography << std::endl;
        //        std::cout << "H actual:"   << homography << std::endl;
        //        std::cout << "H error:"    << error << std::endl;
        //        std::cout << "R error:"    << s.reprojectionError(0) << ";"
        //                                   << s.reprojectionError(1) << ";"
        //                                   << s.reprojectionError(2) << ";"
        //                                   << s.reprojectionError(3) << std::endl;
        //
        //        cv::Mat matchesImg;
        //        cv::drawMatches(transformedImage,
        //                        resKpReal,
        //                        gray,
        //                        sourceKp,
        //                        correctMatches,
        //                        matchesImg,
        //                        cv::Scalar::all(-1),
        //                        cv::Scalar::all(-1),
        //                        std::vector<char>(),
        //                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //
        //        cv::imshow("Matches", matchesImg);
        //        cv::waitKey(-1);
        //    }
        //}
    }

    return true;
}

cv::Scalar computeReprojectionError(const Keypoints& source, const Keypoints& query, const Matches& matches, const cv::Mat& homography)
{
    assert(matches.size() > 0);

    const int pointsCount = matches.size();
    std::vector<cv::Point2f> srcPoints, dstPoints;
    std::vector<float> distances;

    for (int i = 0; i < pointsCount; i++)
    {
        srcPoints.push_back(source[matches[i].trainIdx].pt);
        dstPoints.push_back(query[matches[i].queryIdx].pt);
    }

    cv::perspectiveTransform(dstPoints, dstPoints, homography.inv());
    for (int i = 0; i < pointsCount; i++)
    {
        const cv::Point2f& src = srcPoints[i];
        const cv::Point2f& dst = dstPoints[i];

        cv::Point2f v = src - dst;
        distances.push_back(sqrtf(v.dot(v)));
    }


    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);

    cv::Scalar result;
    result(0) = mean(0);
    result(1) = dev(0);
    result(2) = *std::max_element(distances.begin(), distances.end());
    result(3) = *std::min_element(distances.begin(), distances.end());
    return result;
}
