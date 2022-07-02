
#ifndef PERFORMANCEANALYZER_HPP
#define PERFORMANCEANALYZER_HPP

class PerformanceAnalyzer
{
    public:
        PerformanceAnalyzer();

        // Starts a new timer for generating boards.
        void startBoardGenTimer();

        // Ends the timer for generating boards.
        // Takes the number of boards generated during this time.
        void endBoardGenTimer(unsigned int);

        // Prints the performance report to cout.
        void printReport() const;

    private:
        float getMillis() const;

        float totalGenTime;
        float timerStarted;
        unsigned int totalGenBoard;
};

#endif
