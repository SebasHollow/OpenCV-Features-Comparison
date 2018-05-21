#ifndef ImageTransformation_hpp
#define ImageTransformation_hpp

#include <opencv2/opencv.hpp>
#include <utility>

typedef std::vector<cv::KeyPoint> Keypoints;
typedef cv::Mat                   Descriptors;
typedef std::vector<cv::DMatch>   Matches;

class ImageTransformation
{
public:
    std::string name;
    std::vector<float> m_args;

    virtual std::vector<float> getX() const { return m_args; }

    virtual void transform (float t, const cv::Mat& source, cv::Mat& result) const = 0;

    virtual bool multiplyHomography() const;
    virtual void transform (float t, const Keypoints& source, Keypoints& result) const;
    virtual cv::Mat getHomography (float t, const cv::Mat& source) const;
    virtual ~ImageTransformation();

    static bool findHomography( const Keypoints& source, const Keypoints& result, const Matches& input, Matches& inliers, cv::Mat& homography);

protected:
    ImageTransformation (std::string transformationName)
        {
        name = transformationName;
        }
};

class ImageRotationTransformation : public ImageTransformation
    {
public:
    ImageRotationTransformation (float startAngleInDeg, float endAngleInDeg, float step, const cv::Point2f& rotationCenterInUnitSpace, std::string transformationName = "Rotation");
    ImageRotationTransformation (std::vector<float> angleArgs, std::string transformationName = "Rotation");

    void transform (float t, const cv::Mat& source, cv::Mat& result) const override;
    cv::Mat getHomography (float t, const cv::Mat& source) const override;

private:
    cv::Point2f m_rotationCenterInUnitSpace;
    };

class ImageYRotationTransformation : public ImageTransformation
    {
public:
    ImageYRotationTransformation (float startAngleInDeg, float endAngleInDeg, float step, std::string transformationName = "YRotation");

    void transform (float t, const cv::Mat& source, cv::Mat& result)const override;

    cv::Mat getHomography (float t, const cv::Mat& source) const override;
    bool multiplyHomography() const override;
    };

class ImageXRotationTransformation : public ImageTransformation
    {
public:
    ImageXRotationTransformation (float startAngleInDeg, float endAngleInDeg, float step, std::string trasnformationName = "XRotation");

    void transform (float t, const cv::Mat& source, cv::Mat& result) const override;
    cv::Mat getHomography (float t, const cv::Mat& source) const override;
    bool multiplyHomography() const override;
    };

class ImageScalingTransformation : public ImageTransformation
{
public:
    ImageScalingTransformation (float minScale, float maxScale, float step, std::string transformationName = "Scaling");
    ImageScalingTransformation (std::vector<float> scalingArgs, std::string transformationName = "Scaling");

    void transform(float t, const cv::Mat& source, cv::Mat& result)const override;

    cv::Mat getHomography(float t, const cv::Mat& source) const override;
};

class GaussianBlurTransform : public ImageTransformation
    {
public:
    GaussianBlurTransform (int startSize, int maxKernelSize, int stepSize, std::string transformationName = "Gaussian blur");
    GaussianBlurTransform (std::vector<float> kernelSizeArgs, std::string transformationName = "Gaussian blur");

    void transform (float t, const cv::Mat& source, cv::Mat& result) const override;
    };

class BrightnessTransform : public ImageTransformation
{
public:
    BrightnessTransform (int min, int max, int step, std::string transformationName = "Brightness");
    BrightnessTransform (std::vector<float> intensityArgs, std::string transformationName = "Brightness");

    void transform (float t, const cv::Mat& source, cv::Mat& result) const override;
};

class PerspectiveTransform : public ImageTransformation
    {
public:
    PerspectiveTransform (int count, std::string transformationName = "Perspective");

    void transform (float t, const cv::Mat& source, cv::Mat& result) const override;
    cv::Mat getHomography (float t, const cv::Mat& source) const override;
    void GetPerspectiveTransformationMatrix(const cv::Mat& input, cv::Mat& output, double alpha, double beta,
                                               double gamma, double dx, double dy, double dz, double f) const;

private:
    static cv::Mat warpPerspectiveRand(cv::RNG& rng);

    std::vector<cv::Mat> m_homographies;
    };

class CombinedTransform : public ImageTransformation
{
public:
    typedef enum
    {
        // Generate resulting X vector as list of all possible combinations of first and second args
        Full,
        
        // Largest argument vector used as is, the values for other vector is copied
        Extrapolate,
        
        // Smallest argument vector used as is, the values for other vector is interpolated from other
        Interpolate
    } ParamCombinationType;

    std::string name;
    CombinedTransform (cv::Ptr<ImageTransformation> first, const cv::Ptr<ImageTransformation>& second, ParamCombinationType type = Extrapolate);

    std::vector<float> getX() const override;

    void transform (float t, const cv::Mat& source, cv::Mat& result) const override;
    bool multiplyHomography() const override;
    void transform (float t, const Keypoints& source, Keypoints& result) const override;
    cv::Mat getHomography (float t, const cv::Mat& source) const override;
    
private:
    std::vector<float> m_x;
    std::vector<std::pair<float, float>> m_params;

    cv::Ptr<ImageTransformation> m_first;
    cv::Ptr<ImageTransformation> m_second;
};

#endif