#include "CollectedStatistics.hpp"

#include <sstream>
#include <numeric>
#include <cassert>

template<typename T>
std::string quote(const T& t)
{
    std::ostringstream quoteStr;
    quoteStr << "\"" << t << "\"";
    return quoteStr.str();
}

std::ostream& tab(std::ostream& str)
{
    return str << "\t";
}

std::ostream& null(std::ostream& str)
{
    return str << "NULL";
}

FrameMatchingStatistics::FrameMatchingStatistics()
{
    totalKeypoints = 0;
    argumentValue = 0;
    percentOfMatches = 0;
    ratioTestFalseLevel = 0;
    meanDistance = 0;
    stdDevDistance = 0;
    matchingRatio = 0;
    recall = 0;
    precision = 0;
    consumedTimeMs = 0;
    homographyError = std::numeric_limits<float>::max();
    isValid = false;
    memoryAllocated = 0;
    alg = "";
    trans = "";
}


bool FrameMatchingStatistics::tryGetValue(StatisticElement element, float& value) const
{
    if (!isValid)
        return false;

    switch (element)
    {
    case  StatisticsElementPointsCount:
        value = totalKeypoints;
        return true;

    case StatisticsElementPercentOfCorrectMatches:
        //value = correctMatchesPercent * 100;
        return false;

    case StatisticsElementPercentOfMatches:
        value = percentOfMatches * 100;
        return true;

    case StatisticsElementMeanDistance:
        value = meanDistance;
        return true;

    case StatisticsElementHomographyError:
        value = homographyError;
        return true;

    case StatisticsElementMatchingRatio:
        value = matchingRatio;
        return true;

    case StatisticsElementPatternLocalization:
        //value = patternLocalization();
        return false;
    case StatisticsElementPrecision:
        value = precision;
        return true;
    case StatisticsElementMemoryAllocated:
        value = memoryAllocated;
        return true;
    case StatisticsElementConsumedTimeMs:
        value = consumedTimeMs;
        return true;
    case StatisticsElementConsumedTimeMsPerDescriptor:
        value = consumedTimeMs / totalKeypoints;
        return true;
    case StatisticsElementMemoryAllocatedPerDescriptor:
        value = memoryAllocated / totalKeypoints;
        return true;
    case StatisticsElementRecall:
        value = recall;
        return true;
    default:
        return false;
    }
}

void FrameMatchingStatistics::getAlgTransInfo(std::string& alg, std::string& trans) const {
    alg = this->alg;
    trans = this->trans;
}

std::ostream& FrameMatchingStatistics::writeElement(std::ostream& str, StatisticElement elem) const
{
    float value;
    std::string alg, trans;
    getAlgTransInfo(alg, trans);

    if (tryGetValue(elem, value))
        str << alg << tab << trans << tab << value << std::endl;
    else
        str << alg << tab << trans << tab << null << std::endl;

    return str;
}

SingleRunStatistics& CollectedStatistics::getStatistics(std::string algorithmName, std::string transformationName)
    {
    return m_allStats[std::make_pair(algorithmName, transformationName)];
    }


CollectedStatistics::OuterGroup CollectedStatistics::groupByAlgorithmThenByTransformation() const
    {
    OuterGroup result;

    for (const auto& m_allStat : m_allStats)
        result[m_allStat.first.first][m_allStat.first.second] = &(m_allStat.second);

    return result;
    }

CollectedStatistics::OuterGroupLine CollectedStatistics::groupByTransformationThenByAlgorithm() const
{
    OuterGroup result;

    for (const auto& m_allStat : m_allStats)
        result[m_allStat.first.second][m_allStat.first.first] = &(m_allStat.second);

    OuterGroupLine line;

    for (OuterGroup::const_iterator tIter = result.begin(); tIter != result.end(); ++tIter)
        {
        const std::string transformationName = tIter->first;
        const InnerGroup& inner = tIter->second;

        GroupedByArgument& lineStat = line[transformationName];

        std::vector<const SingleRunStatistics*> statitics;

        for (const auto& algIter : inner)
            {
            const std::string algName = algIter.first;

            lineStat.algorithms.push_back(algName);
            statitics.push_back(algIter.second);
            }

        const SingleRunStatistics& firstStat = *statitics.front();
        const int argumentsCount = firstStat.size();

        for (int i = 0; i < argumentsCount; i++)
        {
            Line l;
            l.argument = firstStat[i].argumentValue;

            for (int algIndex = 0; algIndex < statitics.size(); algIndex++)
            {
                const SingleRunStatistics& s = *statitics[algIndex];

                l.stats.push_back(&s[i]);
            }

            lineStat.lines.push_back(l);
        }
    }


    return line;
}

std::ostream& CollectedStatistics::printAverage(std::ostream& str, StatisticElement elem) const
{
    OuterGroup result;
    str << "Average" << std::endl;

    for (std::map<Key, SingleRunStatistics>::const_iterator i = m_allStats.begin(); i != m_allStats.end(); ++i)
    {
        result[i->first.second][i->first.first] = &(i->second);

        str << i->first.first << tab << i->first.second << tab << average(i->second, elem) << std::endl;
    }

    return str;
}

std::ostream& CollectedStatistics::printStatistics(std::ostream& str, StatisticElement elem) const
{
    OuterGroupLine report = groupByTransformationThenByAlgorithm();

    for (OuterGroupLine::const_iterator tIter = report.begin(); tIter != report.end(); ++tIter)
    {
        const GroupedByArgument& inner = tIter->second;
        for (size_t i = 0; i < inner.lines.size(); i++)
        {
            const Line& l = inner.lines[i];

            for (size_t j = 0; j < l.stats.size(); j++)
            {
                str << l.argument << tab;
                const FrameMatchingStatistics& item = *l.stats[j];
                item.writeElement(str, elem);
            }
        }
    }

    return str << std::endl;
}

std::ostream& CollectedStatistics::printPerformanceStatistics(std::ostream& str) const
{
    str << quote("Performance")               << std::endl;
    str << quote("Algorithm")                 << tab
        << quote("Average time per Frame")    << tab
        << quote("Average time per KeyPoint") << std::endl;

    OuterGroup report = groupByAlgorithmThenByTransformation();

    for (OuterGroup::const_iterator alg = report.begin(); alg != report.end(); ++alg)
    {
        std::vector<double> timePerFrames;
        std::vector<double> timePerKeyPoint;

        for (InnerGroup::const_iterator tIter = alg->second.begin(); tIter != alg->second.end(); ++tIter)
        {
            const SingleRunStatistics& runStatistics = *tIter->second;
            for (size_t i = 0; i < runStatistics.size(); i++)
            {
                if (runStatistics[i].isValid)
                {
                    timePerFrames.push_back(runStatistics[i].consumedTimeMs);
                    timePerKeyPoint.push_back(runStatistics[i].totalKeypoints > 0 ? (runStatistics[i].consumedTimeMs / runStatistics[i].totalKeypoints) : 0);
                }
            }
        }

        double avgPerFrame    = std::accumulate(timePerFrames.begin(),   timePerFrames.end(), 0.0)     / timePerFrames.size();
        double avgPerKeyPoint = std::accumulate(timePerKeyPoint.begin(), timePerKeyPoint.end(), 0.0) / timePerKeyPoint.size();

        str << quote(alg->first) << tab
            << avgPerFrame       << tab
            << avgPerKeyPoint    << std::endl;
    }

    return str << std::endl;
}

float average(const SingleRunStatistics& statistics, StatisticElement element)
{
    std::vector<float> scores;

    for (size_t i = 0; i < statistics.size(); i++)
    {
        float value;
        bool valid = statistics[i].tryGetValue(element, value);

        if (valid)
        {
            scores.push_back(value);
        }
    }

    float sum     = std::accumulate(scores.begin(), scores.end(), 0.0f);
    float average = sum / scores.size();
    return average;
}

float maximum(const SingleRunStatistics& statistics, StatisticElement element)
{
    std::vector<float> scores;

    for (size_t i = 0; i < statistics.size(); i++)
    {
        float value;
        bool valid = statistics[i].tryGetValue(element, value);

        if (valid)
        {
            scores.push_back(value);
        }
    }

    assert(!scores.empty());

    float max = *std::max_element(scores.begin(), scores.end());
    return max;
}

