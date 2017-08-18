#include "FeatureAlgorithm.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <cassert>

static cv::Ptr<cv::flann::IndexParams> indexParamsForDescriptorType(int descriptorType, int defaultNorm)
{
    switch (defaultNorm)
    {
    case cv::NORM_L2:
        return cv::Ptr<cv::flann::IndexParams>(new cv::flann::KDTreeIndexParams());

    case cv::NORM_HAMMING:
        return cv::Ptr<cv::flann::IndexParams>(new cv::flann::LshIndexParams(20, 15, 2));

    default:
        CV_Assert(false && "Unsupported descriptor type");
    };
}

cv::Ptr<cv::DescriptorMatcher> matcherForDescriptorType(int descriptorType, int defaultNorm, bool bruteForce)
{
    if (bruteForce) {
        return cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(defaultNorm, true));
    }
    else {
        return  cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher(indexParamsForDescriptorType(descriptorType, defaultNorm)));
    }
}

FeatureAlgorithm::FeatureAlgorithm(const std::string& n, cv::Ptr<cv::Feature2D> fe, bool useBruteForceMather)
: name(n)
, knMatchSupported(false)
, featureEngine(fe)
, matcher(matcherForDescriptorType(fe->descriptorSize(), fe->defaultNorm(), useBruteForceMather))
{
    CV_Assert(fe);
}


bool FeatureAlgorithm::extractFeatures(const cv::Mat& image, Keypoints& kp, Descriptors& desc) const
{
    assert(!image.empty());
    cv::Ptr<cv::Feature2D> surf_detector = cv::xfeatures2d::SURF::create();
    surf_detector->detect(image, kp);

    if (kp.empty())
        return false;

    featureEngine->compute(image, kp, desc);

    return kp.size() > 0;
}

bool FeatureAlgorithm::extractFeatures(const cv::Mat& image, Keypoints& kp, Descriptors& desc, int64& start, int64& end, size_t& memoryAllocated) const
{
    assert(!image.empty());
    cv::Ptr<cv::Feature2D> surf_detector = cv::xfeatures2d::SURF::create();
    surf_detector->detect(image, kp);

    if (kp.empty())
        return false;

    start = cv::getTickCount();
    //cv::clearMemoryAllocated(); // Only works with custom compiled OpenCV version
    featureEngine->compute(image, kp, desc);
    //memoryAllocated = cv::getAmountOfMemoryAllocated(); // Only works with custom compiled OpenCV version
    end = cv::getTickCount();

    return kp.size() > 0;
}

Descriptors FeatureAlgorithm::getDescriptors(const cv::Mat& image, Keypoints& kp) const
{
    Descriptors desc;
    featureEngine->compute(image, kp, desc);
    return desc;
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, Matches& matches) const
{
    matcher->match(query, train, matches);
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, int k, std::vector<Matches>& matches) const
{
    assert(knMatchSupported);
    matcher->knnMatch(query, train, matches, k);
}

