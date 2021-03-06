#ifndef CollectedStatistics_hpp
#define CollectedStatistics_hpp

#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>

typedef enum
{
    StatisticsElementPointsCount,
    StatisticsElementPercentOfCorrectMatches,
    StatisticsElementPercentOfMatches,
    StatisticsElementMeanDistance,
    StatisticsElementMatchingRatio,
    StatisticsElementHomographyError,
    StatisticsElementPatternLocalization,
    StatisticsElementAverageReprojectionError,
    StatisticsElementRecall,
    StatisticsElementPrecision,
    StatisticsElementMemoryAllocated,
    StatisticsElementConsumedTimeMs,
    StatisticsElementConsumedTimeMsPerDescriptor,
    StatisticsElementMemoryAllocatedPerDescriptor
} StatisticElement;

struct FrameMatchingStatistics
{
    FrameMatchingStatistics();

    std::string alg;
    std::string trans;

    int totalKeypoints;

    float argumentValue;
    float percentOfMatches;
    float ratioTestFalseLevel;
    float meanDistance;
    float stdDevDistance;
    float matchingRatio;
    float homographyError;
    size_t memoryAllocated;

    float recall;
    float precision;

    float consumedTimeMs;
    cv::Scalar reprojectionError;
    bool   isValid;

    // inline float matchingRatio()       const { return matchingRatio * percentOfMatches * 100.0f; };
    // inline float patternLocalization() const { return matchingRatio * percentOfMatches * (1.0f - homographyError); }

    std::ostream& writeElement(std::ostream& str, StatisticElement elem) const;
    void getAlgTransInfo(std::string& alg, std::string& trans) const;
    bool tryGetValue(StatisticElement element, float& value) const;
};

typedef std::vector<FrameMatchingStatistics> SingleRunStatistics;

float average(const SingleRunStatistics& statistics, StatisticElement element);
float maximum(const SingleRunStatistics& statistics, StatisticElement element);

struct Line
{
    float                                 argument;
    std::vector<const FrameMatchingStatistics*> stats;
};

struct GroupedByArgument
{
    std::vector<std::string> algorithms;
    std::vector<Line>        lines;
};

class CollectedStatistics
{
public:
    typedef std::map<std::string, const SingleRunStatistics*> InnerGroup;
    typedef std::map<std::string, InnerGroup>                 OuterGroup;
    typedef std::map<std::string, GroupedByArgument>          OuterGroupLine;

    SingleRunStatistics& getStatistics(std::string algorithmName, std::string transformationName);

    OuterGroup groupByAlgorithmThenByTransformation() const;
    OuterGroupLine groupByTransformationThenByAlgorithm() const;

    std::ostream& printPerformanceStatistics(std::ostream& str) const;
    std::ostream& printStatistics(std::ostream& str, StatisticElement elem) const;
    std::ostream& printAverage(std::ostream& str, StatisticElement elem) const;

private:
    typedef std::pair<std::string, std::string> Key;

    std::map<Key, SingleRunStatistics> m_allStats;
};

#endif
