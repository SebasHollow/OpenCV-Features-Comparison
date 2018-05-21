#ifndef AlgorithmEstimation_hpp
#define AlgorithmEstimation_hpp

#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"
#include "ImageTransformation.hpp"

static bool SAVE_TRANSFORMED_IMAGES = true;

int CountVisibleFeatures (std::vector<cv::Point2f>& sourcePoints, int imageCols, int imageRows);

int CountCorrectMatches(Matches& matches, std::vector<cv::Point2f>& sourcePointsInFrame, Keypoints& resKpReal);

bool computeMatchesDistanceStatistics (const Matches& matches, float& meanDistance, float& stdDev);

void ratioTest (const std::vector<Matches>& knMatches, float maxRatio, Matches& goodMatches);

bool performEstimation (const FeatureAlgorithm& alg,
                        const ImageTransformation& transformation,
                        const cv::Mat& sourceImage,
                        const Keypoints& sourceKp,
                        const Descriptors& sourceDesc,
                        SingleRunStatistics& stat);

#endif
