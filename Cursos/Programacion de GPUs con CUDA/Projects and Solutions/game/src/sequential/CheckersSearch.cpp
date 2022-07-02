
#include <float.h>
#include <cassert>
#include <queue>

#include "CheckersSearch.hpp"

using namespace std;

CheckersSearch::CheckersSearch()
{
    this->pf = new PerformanceAnalyzer();
}

CheckersSearch::~CheckersSearch()
{
    delete this->pf;
}

CheckersBoard CheckersSearch::search(CheckersBoard* board, player p, int ttl) const
{
    CheckersBoard nextBoard; 
    float maxUtil = maxValue(board, &nextBoard, p, FLT_MAX * -1, FLT_MAX, ttl);
    cout << maxUtil << endl;
    return nextBoard;
}

float CheckersSearch::maxValue(CheckersBoard* board, CheckersBoard* out, player p, float a, float b, int ttl) const
{
    if(ttl == 0 || isTerminal(board, p)) return utility(board, p);

    vector<CheckersBoard>* outboards = new vector<CheckersBoard>();
    nextStates(*board, outboards, p);

    // If outboards.size() == 0, we lost.
    // We skip the loop and return FLT_MAX * -1,
    // which is the lowest utility possible.

    // Lookin for maximum, thus starting at lowest possible value.
    float localMaximum = FLT_MAX * -1;
    CheckersBoard localMaximumBoard;
    for(unsigned int i = 0; i < outboards->size(); i++)
    {
        float minUtil = minValue(&outboards->at(i), &localMaximumBoard, p, a, b, ttl-1);
        if(minUtil > localMaximum)
            // Higher than maximum, thus replace maximum.
        {
            localMaximum = minUtil;
            *out = outboards->at(i);
        }
        // Maximum is greater than worst-case global minimum,
        // So minValue will never pick this branch.
        if(localMaximum >= b)
        {
            break;
        }
        // Update a to hgihest of local and global maximum.
        a = max(a, localMaximum);
    }

    vector<CheckersBoard>().swap(*outboards);

    return localMaximum;
}

float CheckersSearch::minValue(CheckersBoard* board, CheckersBoard* out, player p, float a, float b, int ttl) const
{
    if(ttl == 0 || isTerminal(board, !p)) return utility(board, p);

    vector<CheckersBoard>* outboards = new vector<CheckersBoard>();
    nextStates(*board, outboards, !p);

    // If outboards.size() == 0, we won.
    // We skip the loop and return FLT_MAX,
    // which is the highest utility possible.

    // Looking for minimum, thus starting at highest possible value.
    float localMinimum = FLT_MAX;
    CheckersBoard localMinimumBoard;
    for(unsigned int i = 0; i < outboards->size(); i++)
    {
        float maxUtil = maxValue(&outboards->at(i), &localMinimumBoard, p, a, b, ttl-1);
        if(maxUtil < localMinimum)
            // Lower than mimimum, thus replace minimum.
        {
            localMinimum = maxUtil;
            *out = outboards->at(i);
        }
        // Minimum is lower than worst-case global maximum.
        // So maxValue will never pick this branch.
        if(localMinimum <= a)
        {
            break;
        }
        // Update b to lowest of local and global minimum.
        b = min(b, localMinimum);
    }

    vector<CheckersBoard>().swap(*outboards);

    return localMinimum;
}

bool CheckersSearch::isTerminal(CheckersBoard* board, player p) const
{
    if(p == player1) // We are player 1.
    {
        return board->p1NumPieces() == 0;
    }
    else // We are player 2.
    {
        return board->p2NumPieces() == 0;
    }

}

void CheckersSearch::nextStates(CheckersBoard inboard,
        vector<CheckersBoard>* outboards,
        player p) const
{
    this->pf->startBoardGenTimer();
#ifdef DEBUG
    assert(outboards->size() == 0);
#endif
    nextBoardsByTakeover(inboard, outboards, p);
    if(outboards->size() <= 0)
    {
        // We put all new boards that only include regular moves into outboads. 
        nextBoardsByMove(inboard, outboards, p);
    }
    this->pf->endBoardGenTimer(outboards->size());
}

void CheckersSearch::nextBoardsByTakeover(CheckersBoard inboard,
        vector<CheckersBoard>* outboards,
        player p) const
{
#ifdef DEBUG
    assert(outboards->size() == 0);
#endif

    // Load all boards into a queue.
    queue<CheckersBoard>* q = new queue<CheckersBoard>();
    queue<int>* qPiece = new queue<int>();
    q->push(inboard);
    qPiece->push(-1);
    size_t iteration = 0;
    // While there are boards in the queue.
    while(q->size() > 0)
    {
        // Pop a board from the queue.
        CheckersBoard currentBoard = q->front();
        q->pop();
        int chainPieceIndex = qPiece->front();
        qPiece->pop();
        int pieces[12];
        // findPieceIndices places the indices for every piece of the given 
        // player in the pieces array,
        // and returns the number of pieces of the given player.
        int numpieces = findPieceIndices(&currentBoard, pieces, p);
        bool didTakeOver = false;
        // Check possible takeovers for every piece on this board.
        for(int i = 0; i < numpieces; i++)
        {
            int piece = pieces[i];
            if(chainPieceIndex >= 0 && chainPieceIndex != piece) continue;
            int takeovers[4];
            int numTakeovers = possibleTakeovers(&currentBoard, piece, takeovers, p);
            for(int j = 0; j < numTakeovers; j++)
            {
#ifdef DEBUG
                assert(takeovers[j] >= 0 && takeovers[j] < 64);
#endif
                // If we do a takeover, generate the new boards that reflect these actions.
                CheckersBoard *newBoard = new CheckersBoard(currentBoard);
                int dst = takeovers[j];
                int takeover = (takeovers[j] + piece)/2;
                newBoard->captureMove(dst, takeover, piece);
                if(!currentBoard.moveWillKing(dst, piece))
                {
                  q->push(*newBoard);
                  qPiece->push(dst);
                  didTakeOver = true;
                }  
            }
        }
        if(!didTakeOver && iteration > 0)
            // If we cannot do another takeover, and this is not a board
            // from the inboards, we have completed a takeover move
            // which we write to the output.
        {
            outboards->push_back(currentBoard);
        }
        iteration++;
    }
    delete q;
    delete qPiece;
}

void CheckersSearch::nextBoardsByMove(CheckersBoard inboard,
        vector<CheckersBoard>* outboards,
        player p) const
{
#ifdef DEBUG
    assert(outboards->size() == 0);
#endif

    CheckersBoard currentBoard = inboard;
    int pieces[12];
    // findPieceIndices places the indices for every piece of the given 
    // player in the pieces array,
    // and returns the number of pieces of the given player.
    int numpieces = findPieceIndices(&currentBoard, pieces, p);
#ifdef DEBUG
    assert(numpieces > 0);
#endif
    // Check possible takeovers for every piece on this board.
    for(int j = 0; j < numpieces; j++)
    {
        int piece = pieces[j];
        int moves[4];
        int numMoves = possibleRegularMoves(&currentBoard, piece, moves, p);
        for(int k = 0; k < numMoves; k++)
        {
#ifdef DEBUG
            assert(moves[k] >= 0 && moves[k] < 64);
#endif
            // If we do a takeover, generate the new boards that reflect these actions.
            CheckersBoard *newBoard = new CheckersBoard(currentBoard);
            newBoard->move(moves[k], piece);
            outboards->push_back(*newBoard);
        }
    }
}

int CheckersSearch::findPieceIndices(CheckersBoard* board,
        int* pieces,
        player p) const
{
    int foundPieces = 0;
    int piecesIndex = 0;
    for(int i = 0; i < 64; i++)
    {
        if(board->isOccupiableSpace(i))
        {
            if(p == player1)
            {
                if(board->p1Piece(i))
                {
                    foundPieces++;
                    pieces[piecesIndex++] = i;
                }
            }
            else
            {
                if(board->p2Piece(i))
                {
                    foundPieces++;
                    pieces[piecesIndex++] = i;
                }
            }
        }
    }
    return foundPieces;
}

int CheckersSearch::possibleTakeovers(CheckersBoard* board,
        int pieceIndex,
        int* targets,
        player p) const
{
    return possibleMoves(board, pieceIndex, targets, p, 14, 18);
}

int CheckersSearch::possibleRegularMoves(CheckersBoard* board,
        int pieceIndex,
        int* targets,
        player p) const
{
    return possibleMoves(board, pieceIndex, targets, p, 7, 9);
}

int CheckersSearch::possibleMoves(CheckersBoard* board,
        int pieceIndex,
        int* targets,
        player p,
        int jump1,
        int jump2) const
{
    int numMoves = 0;
    int moves[] = { pieceIndex - jump1, pieceIndex - jump2,
        pieceIndex + jump1, pieceIndex + jump2 };
    int start = 0;
    int end = 4;
    if(board->p1King(pieceIndex) || board->p2King(pieceIndex))
    {
        // Check all four possible moves.
    }
    else if(p == player1)
    {
        // Check the minus moves.
        start = 2;
    }
    else if(p == player2)
    {
        // Check the plus moves.
        end = 2;
    }

    for(int i = start; i < end; i++)
    {
        if(board->isOccupiableSpace(moves[i]))
        {
            if( board->canMove(moves[i], pieceIndex, p))
            {
                targets[numMoves++] = moves[i];
            }
        }
    }

    return numMoves;
}

float CheckersSearch::utility(CheckersBoard* board, player p) const
{
  float res = 0;
  int sign = 0;

  if(p) sign = 1;
  else  sign = -1;

#ifdef DEBUG
  assert(sign != 0);
#endif

  // max value here is if every man is a king = 12*15 = 180, and the other 
  // player has no pieces = 0
  size_t p1PieceVal = board->p1WeightedMen() + board->p1NumKings() * 15;
  size_t p2PieceVal = board->p2WeightedMen() + board->p2NumKings() * 15;
  const float scoreMultiplier = int(std::numeric_limits<int>::max()) / int(180);
  if (p1PieceVal > p2PieceVal) {
      res = int((float(p1PieceVal) / float(p2PieceVal)) * scoreMultiplier);
  }
  else if (p1PieceVal < p2PieceVal) {
      res = int((-1.0f * float(p1PieceVal) / float(p2PieceVal)) * scoreMultiplier);
  }

  return sign * res;
}

void CheckersSearch::utility(vector<float>* utilities, vector<CheckersBoard>* boards, player p) const
{
    for(unsigned int i = 0; i < boards->size(); i++)
    {
        utilities->push_back(utility(&boards->at(i), p));
    }
}

CheckersBoard* CheckersSearch::argmaxUtility(vector<CheckersBoard>* boards, player p) const
{
#ifdef DEBUG
    assert(boards->size() > 0);
#endif

    vector<float> utilities;
    utility(&utilities, boards, p);

    float localMax = utilities[0];
    CheckersBoard* localMaxBoard = &boards->at(0);
    for(unsigned int i = 1; i < boards->size(); i++)
    {
        if(utilities[i] > localMax)
        {
            localMax = utilities[i];
            localMaxBoard = &boards->at(i);
        }
    }

#ifdef DEBUG
    assert(localMax != FLT_MIN);
#endif
    return localMaxBoard;
}

