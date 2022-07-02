
#include "HumanPlayer.hpp"

#include "CheckersSearch.hpp"

CheckersBoard HumanPlayer::makeMove(CheckersBoard board, player p) const
{
    // Ask for keyboard input to make move. 
    CheckersBoard newBoard = CheckersBoard(board);
    cout << "Enter index from piece to move, followed by its destinations." <<
     endl;
    int pieceindex;
    int destindex;
    cin >> pieceindex;
    cin >> destindex;

    if(board.isOccupiableSpace(pieceindex)
            && board.isOccupiableSpace(destindex))
    {
        if(board.canMove(destindex, pieceindex, p))
        {
            newBoard.move(destindex, pieceindex);
        }
    }

    return newBoard;
}
