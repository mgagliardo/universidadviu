#ifndef CHECKERSSEARCH_HPP
#define CHECKERSSEARCH_HPP

#include <vector>

#include "checkersboard.cpp"
#include "PerformanceAnalyzer.hpp"

typedef bool player;

const player player1 = true;
const player player2 = false;

class CheckersSearch
{
    public:
        CheckersSearch();
        ~CheckersSearch();

        CheckersBoard search(CheckersBoard*, player, int) const;

        // Checks if a the given board is in a terminal state for the given player.
        bool isTerminal(CheckersBoard*, player) const;

        PerformanceAnalyzer* pf;

    private:
        // The recursive methods that searches for this players best move.
        float maxValue(CheckersBoard*, CheckersBoard*, player, float, float, int) const;
        float minValue(CheckersBoard*, CheckersBoard*, player, float, float, int) const;

        // Calculates all next possible boards from the given boards.
        void nextStates(CheckersBoard, std::vector<CheckersBoard>*, player) const;

        // Calculates next possible boards that include takeovers.
        void nextBoardsByTakeover(CheckersBoard, std::vector<CheckersBoard>*, player) const;

        // Calculates next possible boards that include regular moves.
        void nextBoardsByMove(CheckersBoard, std::vector<CheckersBoard>*, player) const;

        // Places all indices that correspond with a piece of the given player in the int array.
        // Returns the number of pieces that have been places in the array.
        int findPieceIndices(CheckersBoard*, int*, player) const;

        // Takes a board and a piece-index. Places all destination indices in
        // the output array.
        // These indices correspond with places on the board where the given
        // piece would land on when this piece would do a take-over move.
        // Returns the number of takeovers that have been placed in the array.
        int possibleTakeovers(CheckersBoard*, int, int*, player) const;

        // Takes a board and a piece-index. Places all destination indices in
        // the output array.
        // These indicies correspond with places on the board where the given
        // piece would land on when this piece would do a regular move.
        // Returns the number of regular moves that have been placed in the
        // array.
        int possibleRegularMoves(CheckersBoard*, int, int*, player)
         const;

        // The function that is being used by both possibleTakeovers and
        // possibleMoves.
        // This function tries to move a piece in all four directions, with the
        // given jump size (7/9 for regular moves, 14/18 for takeovers).
        // If the piece can move in one of the directions, the destination index
        // is placed in the given array.
        // Returns the number of moves this piece can make.
        int possibleMoves(CheckersBoard*, int, int*, player, int, int)
         const;

        // Calculates the utility for the given board from the perspective of the given player.
        float utility(CheckersBoard*, player) const;

        // Calculates the utility for all given boards from the perspective of the given player.
        void utility(vector<float>*, std::vector<CheckersBoard>*, player) const;

        // Returns the board from the array of given boards that has the highest utiliy.
        CheckersBoard* argmaxUtility(std::vector<CheckersBoard>*, player) const;
};

#endif
