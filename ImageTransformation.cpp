#include "ImageTransformation.hpp"

bool ImageTransformation::multiplyHomography() const
    {
    return false;
    }

void ImageTransformation::transform (float t, const Keypoints& source, Keypoints& result) const
    {}

cv::Mat ImageTransformation::getHomography(float t, const cv::Mat& source) const
    {
    return cv::Mat::eye (3, 3, CV_64FC1);
    }

ImageTransformation::~ImageTransformation() = default;

bool ImageTransformation::findHomography (const Keypoints& source, const Keypoints& result, const Matches& input, Matches& inliers, cv::Mat& homography)
{
    inliers.clear();

    if (input.size() < 4)
        return false;

    const int pointsCount = input.size();
    const float reprojectionThreshold = 3;

    //Prepare src and dst points
    std::vector<cv::Point2f> srcPoints, dstPoints;
    for (int i = 0; i < pointsCount; i++)
    {
        srcPoints.push_back(source[input[i].trainIdx].pt);
        dstPoints.push_back(result[input[i].queryIdx].pt);
    }

    // Find homography using RANSAC algorithm
    std::vector<unsigned char> status;
    homography = cv::findHomography(srcPoints, dstPoints, cv::LMEDS, reprojectionThreshold, status);

    if (homography.empty() || std::count(status.begin(), status.end(), 1) < 4)
        return false;

    for (int i = 0; i < pointsCount; i++)
    {
        if (status[i])
        {
            inliers.push_back(input[i]);
        }
    }
    return true;

    /*
    // Warp dstPoints to srcPoints domain using inverted homography transformation
    std::vector<cv::Point2f> srcReprojected;
    cv::perspectiveTransform(dstPoints, srcReprojected, homography.inv());

    // Pass only matches with low reprojection error (less than reprojectionThreshold value in pixels)
    inliers.clear();
    for (int i = 0; i < pointsCount; i++)
    {
        cv::Point2f actual = srcPoints[i];
        cv::Point2f expect = srcReprojected[i];
        cv::Point2f v = actual - expect;
        float distanceSquared = v.dot(v);

        if (distanceSquared <= reprojectionThreshold * reprojectionThreshold)
        {
            inliers.push_back(input[i]);
        }
    }

    // Test for bad case
    if (inliers.size() < 4)
        return false;

    // Now use only good points to find refined homography:
    std::vector<cv::Point2f> refinedSrc, refinedDst;
    for (int i = 0; i < inliers.size(); i++)
    {
        refinedSrc.push_back(source[inliers[i].trainIdx].pt);
        refinedDst.push_back(result[inliers[i].queryIdx].pt);
    }

    // Use least squares method to find precise homography
    cv::Mat homography2 = cv::findHomography(refinedSrc, refinedDst, 0, reprojectionThreshold);

    // Reproject again:
    cv::perspectiveTransform(dstPoints, srcReprojected, homography2.inv());
    inliers.clear();

    for (int i = 0; i < pointsCount; i++)
    {
        cv::Point2f actual = srcPoints[i];
        cv::Point2f expect = srcReprojected[i];
        cv::Point2f v = actual - expect;
        float distanceSquared = v.dot(v);

        if (distanceSquared <= reprojectionThreshold * reprojectionThreshold)
        {
            inliers.push_back(input[i]);
        }
    }

    homography = homography2;
    return inliers.size() >= 4;
    */
}

#pragma mark - ImageRotationTransformation implementation

ImageRotationTransformation::ImageRotationTransformation(float startAngleInDeg, float endAngleInDeg, float step, const cv::Point2f& rotationCenterInUnitSpace, std::string transformationName)
    : ImageTransformation (transformationName), m_rotationCenterInUnitSpace (rotationCenterInUnitSpace)
    {
    // Fill the arguments
    for (float arg = startAngleInDeg; arg <= endAngleInDeg; arg += step)
        if (arg != 0)
            m_args.push_back (arg);
    }

ImageRotationTransformation::ImageRotationTransformation (std::vector<float> angleArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    m_args = angleArgs;
    }

void ImageRotationTransformation::transform (float t, const cv::Mat& source, cv::Mat& result) const
    {
    const cv::Point2f center (source.cols / 2, source.rows / 2);

    cv::Mat rotationMat = getRotationMatrix2D (center, -t, 1);
    
    auto cos = abs (rotationMat.at<double>(0, 0));
    auto sin = abs (rotationMat.at<double>(0, 1));

    auto nW = int (source.rows * sin + source.cols * cos);
    auto nH = int (source.rows * cos + source.cols * sin);

    rotationMat.at<double>(0, 2) += nW / 2 - source.cols / 2;
    rotationMat.at<double>(1, 2) += nH / 2 - source.rows / 2;

    warpAffine (source, result, rotationMat, cv::Size (nW, nH));
    }

// void ImageRotationTransformation::transform(float t, const cv::Mat& source, cv::Mat& result) const {
//     cv::Point2f center(source.cols / 2.0, source.rows / 2.0);
//     cv::Mat rot = cv::getRotationMatrix2D(center, t, 1.0);
//     cv::Rect bbox = cv::RotatedRect(center, source.size(), t).boundingRect();
//     rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
//     rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
//     cv::warpAffine(source, result, rot, bbox.size());
// }

cv::Mat ImageRotationTransformation::getHomography (float angle, const cv::Mat& source) const
    {
    const cv::Point2f center (source.cols * m_rotationCenterInUnitSpace.x, source.rows * m_rotationCenterInUnitSpace.y);
    cv::Mat rotationMat = getRotationMatrix2D (center, angle, 1);

    cv::Mat h = cv::Mat::eye (3, 3, CV_64FC1);
    rotationMat.copyTo (h (cv::Range (0, 2), cv::Range (0, 3)));
    return h;
    }

// cv::Mat ImageRotationTransformation::getHomography(float t, const cv::Mat& source) const
// {
//     cv::Point2f center(source.cols * m_rotationCenterInUnitSpace.x, source.rows * m_rotationCenterInUnitSpace.y);
//     cv::Mat rotationMat = cv::getRotationMatrix2D(center, t, 1);
//     cv::Rect bbox = cv::RotatedRect(center, source.size(), t).boundingRect();
//     rotationMat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
//     rotationMat.at<double>(1, 2) += bbox.height / 2.0 - center.y;
//     cv::Mat h = cv::Mat::eye(3, 3, CV_64FC1);
//     rotationMat.copyTo(h(cv::Range(0, 2), cv::Range(0, 3)));
//     return h;
// }

#pragma mark - ImageYRotationTransformation implementation

ImageYRotationTransformation::ImageYRotationTransformation (float startAngleInDeg, float endAngleInDeg, float step, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    // Fill the arguments
    for (float arg = startAngleInDeg; arg <= endAngleInDeg; arg += step)
        if (arg != 0)
            m_args.push_back (arg);
    }

ImageYRotationTransformation::ImageYRotationTransformation (std::vector<float> angleArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    m_args = angleArgs;
    }

void ImageYRotationTransformation::transform (float t, const cv::Mat& source, cv::Mat& result) const
    {
    warpPerspective (source, result, getHomography (t, source), source.size(), cv::INTER_LANCZOS4);
    }

cv::Mat ImageYRotationTransformation::getHomography (float t, const cv::Mat& source) const
    {
    double beta = ((90 - t) - 90.) * CV_PI / 180.;
    double w = (double)source.cols;
    double h = (double)source.rows;
    cv::Mat A1 = (cv::Mat_<double>(4, 3) <<
                  1, 0, -w / 2,
                  0, 1, -h / 2,
                  0, 0,    0,
                  0, 0,    1);
    cv::Mat RY = (cv::Mat_<double>(4, 4) <<
                  cos(beta), 0, -sin(beta), 0,
                  0, 1,          0, 0,
                  sin(beta), 0,  cos(beta), 0,
                  0, 0,          0, 1);
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
                 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, source.cols,
                 0, 0, 0, 1);
    // 3D -> 2D matrix
    cv::Mat A2 = (cv::Mat_<double>(3, 4) <<
                  source.cols, 0, w / 2, 0,
                  0, source.cols, h / 2, 0,
                  0, 0,   1, 0);
    // Final transformation matrix
    cv::Mat trans = A2 * (T * (RY * A1));
    return trans;
    }

bool ImageYRotationTransformation::multiplyHomography() const
    {
    return true;
    }

#pragma mark - ImageXRotationTransformation implementation

ImageXRotationTransformation::ImageXRotationTransformation (float startAngleInDeg, float endAngleInDeg, float step, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    // Fill the arguments
    for (float arg = startAngleInDeg; arg <= endAngleInDeg; arg += step)
        if (arg != 0)
            m_args.push_back (arg);
    }

ImageXRotationTransformation::ImageXRotationTransformation (std::vector<float> angleArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    m_args = angleArgs;
    }

void ImageXRotationTransformation::transform (float t, const cv::Mat& source, cv::Mat& result) const
    {
    warpPerspective (source, result, getHomography (t, source), source.size(), cv::INTER_LANCZOS4);
    }

cv::Mat ImageXRotationTransformation::getHomography(float t, const cv::Mat& source) const
{
    double alpha = ((90 - t) - 90.) * CV_PI / 180.;
    double w = (double)source.cols;
    double h = (double)source.rows;
    cv::Mat A1 = (cv::Mat_<double>(4, 3) <<
                  1, 0, -w / 2,
                  0, 1, -h / 2,
                  0, 0,    0,
                  0, 0,    1);
    cv::Mat RX = (cv::Mat_<double>(4, 4) <<
                  1,          0,           0, 0,
                  0, cos(alpha), -sin(alpha), 0,
                  0, sin(alpha),  cos(alpha), 0,
                  0,          0,           0, 1);
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
                 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, source.rows,
                 0, 0, 0, 1);
    // 3D -> 2D matrix
    cv::Mat A2 = (cv::Mat_<double>(3, 4) <<
                  source.rows, 0, w / 2, 0,
                  0, source.rows, h / 2, 0,
                  0, 0,   1, 0);
    // Final transformation matrix
    cv::Mat trans = A2 * (T * (RX * A1));
    return trans;
}

bool ImageXRotationTransformation::multiplyHomography() const
{
    return true;
}

#pragma mark - ImageScalingTransformation implementation

ImageScalingTransformation::ImageScalingTransformation(float minScale, float maxScale, float step, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    // Fill the arguments
    for (float arg = minScale; arg <= maxScale; arg += step)
        if (arg != 1.0f)
            m_args.push_back (arg);
    }

ImageScalingTransformation::ImageScalingTransformation (std::vector<float> scalingArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {    
    m_args = scalingArgs;
    }

void ImageScalingTransformation::transform (float t, const cv::Mat& source, cv::Mat& result)const
{
    cv::Size dstSize(static_cast<int>(source.cols * t + 0.5f), static_cast<int>(source.rows * t + 0.5f));
    cv::resize(source, result, dstSize, cv::INTER_AREA);
}

cv::Mat ImageScalingTransformation::getHomography (float t, const cv::Mat& source) const
    {
    cv::Mat h = cv::Mat::eye (3, 3, CV_64FC1);
    h.at<double>(0, 0) = h.at<double>(1, 1) = t;
    return h;
    }

#pragma mark - GaussianBlurTransform implementation

GaussianBlurTransform::GaussianBlurTransform (int startSize, int maxKernelSize, int stepSize, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    for (int arg = startSize; arg <= maxKernelSize; arg += stepSize)
        if (arg > 1)
            m_args.push_back (static_cast<float> (arg));
    }

GaussianBlurTransform::GaussianBlurTransform (std::vector<float> kernelSizeArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    m_args = kernelSizeArgs;
    }

void GaussianBlurTransform::transform (float t, const cv::Mat& source, cv::Mat& result)const
    {
    const int kernelSize = static_cast<int>(t) * 2 + 1;
    GaussianBlur (source, result, cv::Size (kernelSize, kernelSize), 0);
    }

#pragma mark - BrightnessImageTransform implementation

BrightnessTransform::BrightnessTransform (int min, int max, int step, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    for (int arg = min; arg <= max; arg += step)
        if (arg != 0)
            m_args.push_back(static_cast<float>(arg));
    }

BrightnessTransform::BrightnessTransform (std::vector<float> intensityArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    m_args = intensityArgs;
    }

void BrightnessTransform::transform (float t, const cv::Mat& source, cv::Mat& result) const
    {
    result = source + cv::Scalar (t, t, t, t);
    }

#pragma mark - CombinedTransform implementation

CombinedTransform::CombinedTransform (const cv::Ptr<ImageTransformation> first, const cv::Ptr<ImageTransformation>& second, ParamCombinationType type)
    : ImageTransformation (first->name + "+" + second->name)
    {
    std::vector<float> x1 = first->getX();
    std::vector<float> x2 = second->getX();

    switch (type)
    {
    case Full:
        {
        int index = 0;
        for (float& i1 : x1)
            for (float& i2 : x2)
                {
                m_params.emplace_back (i1, i2);
                m_x.push_back (index);
                index++;
                }
        }
    break;


    case Interpolate:
    {
        if (x1.size() > x2.size())
        {
            int index = 0;
            for (size_t i2 = 0; i2 < x2.size(); i2++)
            {
                size_t i1 = static_cast<size_t>(static_cast<float>(x1.size() * i2) / static_cast<float>(x2.size()) + 0.5f);
                m_params.emplace_back (x1[i1], x2[i2]);
                m_x.push_back(index);
                index++;
            }
        }
        else
        {
            int index = 0;
            for (size_t i1 = 0; i1 < x1.size(); i1++)
            {
                size_t i2 = static_cast<size_t>(static_cast<float>(x2.size() * i1) / static_cast<float>(x1.size()) + 0.5f);
                m_params.emplace_back (x1[i1], x2[i2]);
                m_x.push_back(index);
                index++;
            }
        }


    }; break;


    case Extrapolate:
    {
        if (x1.size() > x2.size())
        {
            int index = 0;
            for (size_t i1 = 0; i1 < x1.size(); i1++)
            {
                size_t i2 = static_cast<size_t>(static_cast<float>(x2.size() * i1) / static_cast<float>(x1.size()) );
                m_params.emplace_back (x1[i1], x2[i2]);
                m_x.push_back(index);
                index++;
            }
        }
        else
        {
            int index = 0;
            for (size_t i2 = 0; i2 < x2.size(); i2++)
            {
                size_t i1 = static_cast<size_t>(static_cast<float>(x1.size() * i2) / static_cast<float>(x2.size()) );
                m_params.emplace_back (x1[i1], x2[i2]);
                m_x.push_back(index);
                index++;
            }
        }
    }; break;

    default:
        break;
    }
}

std::vector<float> CombinedTransform::getX() const
    {
    return m_x;
    }

void CombinedTransform::transform (float t, const cv::Mat& source, cv::Mat& result) const
{
    size_t index = static_cast<size_t>(t);
    float t1 = m_params[index].first;
    float t2 = m_params[index].second;

    if (!multiplyHomography()) {
        cv::Mat temp;
        m_first->transform(t1, source, temp);
        m_second->transform(t2, temp, result);
    } else {
        cv::Mat first_homography = m_first->getHomography(t1, source);
        cv::Mat second_homography = m_second->getHomography(t2, source);
        cv::Mat combo = first_homography * second_homography;
        cv::warpPerspective(source, result, combo, source.size(), cv::INTER_LANCZOS4);
    }
}

bool CombinedTransform::multiplyHomography() const
{
    return m_first->multiplyHomography() && m_second->multiplyHomography();
}

void CombinedTransform::transform(float t, const Keypoints& source, Keypoints& result) const
{
    size_t index = static_cast<size_t>(t);
    float t1 = m_params[index].first;
    float t2 = m_params[index].second;
    Keypoints temp;
    m_first->transform(t1, source, temp);
    m_second->transform(t2, temp, result);
}

cv::Mat CombinedTransform::getHomography (float t, const cv::Mat& source) const
{
    size_t index = static_cast<size_t>(t);

    float t1 = m_params[index].first;
    float t2 = m_params[index].second;

    if (!multiplyHomography()) {
        cv::Mat temp;
        m_first->transform(t1, source, temp);
        return m_second->getHomography(t2, temp) * m_first->getHomography(t1, source);
    }
    cv::Mat first_homography = m_first->getHomography(t1, source);
    cv::Mat second_homography = m_second->getHomography(t2, source);
    return first_homography * second_homography;
}

#pragma mark PerspectiveTransform implementation

PerspectiveTransform::PerspectiveTransform (int count, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    cv::RNG rng;

    for (int i = 0; i < count; i++)
        {
        m_args.push_back (i);
        m_homographies.push_back (warpPerspectiveRand (rng));
        }
    }

PerspectiveTransform::PerspectiveTransform (std::vector<float> angleArgs, std::string transformationName)
    : ImageTransformation (transformationName)
    {
    m_args = angleArgs;
    }

cv::Mat PerspectiveTransform::warpPerspectiveRand (cv::RNG& rng)
    {
    cv::Mat H;

    H.create(3, 3, CV_64FC1);
    H.at<double>(0, 0) = rng.uniform ( 0.8f, 1.2f);
    H.at<double>(0, 1) = rng.uniform (-0.1f, 0.1f);
    //H.at<double>(0,2) = rng.uniform(-0.1f, 0.1f)*src.cols;
    H.at<double>(0, 2) = rng.uniform (-0.1f, 0.1f);
    H.at<double>(1, 0) = rng.uniform (-0.1f, 0.1f);
    H.at<double>(1, 1) = rng.uniform ( 0.8f, 1.2f);
    //H.at<double>(1,2) = rng.uniform(-0.1f, 0.1f)*src.rows;
    H.at<double>(1, 2) = rng.uniform (-0.1f, 0.1f);
    H.at<double>(2, 0) = rng.uniform (-1e-4f, 1e-4f);
    H.at<double>(2, 1) = rng.uniform (-1e-4f, 1e-4f);
    H.at<double>(2, 2) = rng.uniform (0.8f, 1.2f);

    return H;
    }

void rotateImage (const cv::Mat &input, cv::Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f)
    {
    alpha = (alpha - 90.) * CV_PI / 180.;
    beta = (beta - 90.) * CV_PI / 180.;
    gamma = (gamma - 90.) * CV_PI / 180.;
    // get width and height for ease of use in matrices
    const auto w = static_cast<double>(input.cols);
    const auto h = static_cast<double>(input.rows);
    // Projection 2D -> 3D matrix
    cv::Mat A1 = (cv::Mat_<double>(4, 3) <<
                  1, 0, -w / 2,
                  0, 1, -h / 2,
                  0, 0,    0,
                  0, 0,    1);
    // Rotation matrices around the X, Y, and Z axis
    cv::Mat RX = (cv::Mat_<double>(4, 4) <<
                  1,          0,           0, 0,
                  0, cos(alpha), -sin(alpha), 0,
                  0, sin(alpha),  cos(alpha), 0,
                  0,          0,           0, 1);
    cv::Mat RY = (cv::Mat_<double>(4, 4) <<
                  cos(beta), 0, -sin(beta), 0,
                  0, 1,          0, 0,
                  sin(beta), 0,  cos(beta), 0,
                  0, 0,          0, 1);
    cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
                  cos(gamma), -sin(gamma), 0, 0,
                  sin(gamma),  cos(gamma), 0, 0,
                  0,          0,           1, 0,
                  0,          0,           0, 1);
    // Composed rotation matrix with (RX, RY, RZ)
    cv::Mat R = RX * RY * RZ;
    // Translation matrix
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
                 1, 0, 0, dx,
                 0, 1, 0, dy,
                 0, 0, 1, dz,
                 0, 0, 0, 1);
    // 3D -> 2D matrix
    cv::Mat A2 = (cv::Mat_<double>(3, 4) <<
                  f, 0, w / 2, 0,
                  0, f, h / 2, 0,
                  0, 0,   1, 0);
    // Final transformation matrix
    cv::Mat trans = A2 * (T * (R * A1));
    // Apply matrix transformation

    warpPerspective (input, output, trans, input.size(), cv::INTER_LANCZOS4);
    }

void PerspectiveTransform::transform (float t, const cv::Mat& source, cv::Mat& result) const
    {
    //rotateImage (source, result, 45, 90, 90, 0, 0, source.rows, source.rows);
    rotateImage (source, result, 90, t, 90, 0, 0, source.rows, source.rows);
    }

cv::Mat PerspectiveTransform::getHomography (float t, const cv::Mat& source) const
    {
    cv::Mat result;
    GetPerspectiveTransformationMatrix (source, result, 90, t * 15, 90, 0, 0, source.rows, source.rows);

    cv::Mat h = cv::Mat::eye (3, 3, CV_64FC1);
    result.copyTo (h (cv::Range (0, 2), cv::Range (0, 3)));
    return h;
    }

void PerspectiveTransform::GetPerspectiveTransformationMatrix (const cv::Mat &input, cv::Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f) const
    {
    alpha = (alpha - 90.) * CV_PI / 180.;
    beta = (beta - 90.) * CV_PI / 180.;
    gamma = (gamma - 90.) * CV_PI / 180.;
    // get width and height for ease of use in matrices
    const auto w = static_cast<double>(input.cols);
    const auto h = static_cast<double>(input.rows);
    // Projection 2D -> 3D matrix
    cv::Mat A1 = (cv::Mat_<double>(4, 3) <<
                  1, 0, -w / 2,
                  0, 1, -h / 2,
                  0, 0,    0,
                  0, 0,    1);
    // Rotation matrices around the X, Y, and Z axis
    cv::Mat RX = (cv::Mat_<double>(4, 4) <<
                  1,          0,           0, 0,
                  0, cos(alpha), -sin(alpha), 0,
                  0, sin(alpha),  cos(alpha), 0,
                  0,          0,           0, 1);
    cv::Mat RY = (cv::Mat_<double>(4, 4) <<
                  cos(beta), 0, -sin(beta), 0,
                  0, 1,          0, 0,
                  sin(beta), 0,  cos(beta), 0,
                  0, 0,          0, 1);
    cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
                  cos(gamma), -sin(gamma), 0, 0,
                  sin(gamma),  cos(gamma), 0, 0,
                  0,          0,           1, 0,
                  0,          0,           0, 1);
    // Composed rotation matrix with (RX, RY, RZ)
    cv::Mat R = RX * RY * RZ;
    // Translation matrix
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
                 1, 0, 0, dx,
                 0, 1, 0, dy,
                 0, 0, 1, dz,
                 0, 0, 0, 1);
    // 3D -> 2D matrix
    cv::Mat A2 = (cv::Mat_<double>(3, 4) <<
                  f, 0, w / 2, 0,
                  0, f, h * 2/*/ 2*/, 0,
                  0, 0,   1, 0);
    // Final transformation matrix
    output = A2 * (T * (R * A1));
    }
