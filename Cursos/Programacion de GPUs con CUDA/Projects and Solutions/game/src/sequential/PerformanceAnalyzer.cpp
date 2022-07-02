
#include <iostream>
#include <time.h>

#include "PerformanceAnalyzer.hpp"

using namespace std;

    PerformanceAnalyzer::PerformanceAnalyzer()
: totalGenTime(0), timerStarted(0), totalGenBoard(0)
{
    // Empty
}

void PerformanceAnalyzer::startBoardGenTimer()
{
    // Start a local timer.
    this->timerStarted = getMillis();
}

void PerformanceAnalyzer::endBoardGenTimer(unsigned int totalBoardsGenerated)
{
    // End the timer and add the number of boards to the total.
    this->totalGenTime += getMillis() - this->timerStarted;
    this->timerStarted = 0;
    this->totalGenBoard += totalBoardsGenerated;
}

void PerformanceAnalyzer::printReport() const
{
    cout << "Generated " << this->totalGenBoard / this->totalGenTime
     << " boards/ms." << endl;
}

float PerformanceAnalyzer::getMillis() const
{
    return (float(clock()) / CLOCKS_PER_SEC)*1000;
}

